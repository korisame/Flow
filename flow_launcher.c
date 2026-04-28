// ─────────────────────────────────────────────────────────────────────────────
//  flow_launcher.c — Flow main bundle binary
//
//  Compiled as Flow.app/Contents/MacOS/Flow.  Because THIS binary (not python)
//  is the executable of the bundle, macOS / TCC identifies the process as
//  "Flow" in System Settings → Privacy & Security → Accessibility / Microphone.
//
//  At runtime it:
//    1. Locates ../Resources/ relative to the binary.
//    2. Runs install.sh (creates ~/.flow/venv, pip-installs deps) on first run.
//    3. dlopen()'s libpython3.12 from Homebrew and calls Py_Main(flow.py),
//       so the Python interpreter runs INSIDE this "Flow" process — TCC
//       attribution stays on Flow.
//
//  Build:  clang -O2 -o Flow flow_launcher.c -framework CoreFoundation
// ─────────────────────────────────────────────────────────────────────────────
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include <mach-o/dyld.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <wchar.h>

typedef wchar_t* (*Py_DecodeLocale_fn)(const char*, size_t*);
typedef int      (*Py_Main_fn)(int, wchar_t**);

static int file_exists(const char *p) {
    struct stat st;
    return stat(p, &st) == 0;
}

static int run_installer(const char *resources_dir) {
    char installer[4096];
    snprintf(installer, sizeof(installer), "%s/install.sh", resources_dir);
    if (!file_exists(installer)) {
        fprintf(stderr, "[Flow] install.sh not found at %s\n", installer);
        return 127;
    }
    char cmd[8192];
    snprintf(cmd, sizeof(cmd), "/bin/bash \"%s\"", installer);
    return system(cmd);
}

static void redirect_logs(const char *home) {
    // Always redirect to ~/.flow/app.log so we can debug regardless of how
    // the binary was launched (Finder / open / launchd / shell).
    char logpath[4096];
    snprintf(logpath, sizeof(logpath), "%s/.flow/app.log", home);
    FILE *f = freopen(logpath, "a", stderr);
    if (f) freopen(logpath, "a", stdout);
    setvbuf(stderr, NULL, _IOLBF, 0);
    setvbuf(stdout, NULL, _IOLBF, 0);
}

static void show_alert(const char *title, const char *msg) {
    char cmd[8192];
    snprintf(cmd, sizeof(cmd),
        "/usr/bin/osascript -e 'display alert \"%s\" message \"%s\" as critical'",
        title, msg);
    system(cmd);
}

int main(int argc, char *argv[]) {
    // ── 1. Locate Flow.app/Contents/Resources ───────────────────────────────
    char exe_path[4096];
    uint32_t sz = sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &sz) != 0) {
        fprintf(stderr, "[Flow] Cannot resolve own executable path\n");
        return 1;
    }

    // Refuse to run when launched from a mounted DMG (or any /Volumes/ path).
    // Symptom otherwise: Flow stops the moment the user ejects the DMG.
    if (strncmp(exe_path, "/Volumes/", 9) == 0) {
        show_alert(
            "Install Flow first",
            "Flow is running from the DMG. Close Flow, eject the DMG, "
            "then run \\\"Install Flow.command\\\" inside the DMG to install it "
            "into /Applications properly."
        );
        return 10;
    }
    char exe_copy[4096];
    strncpy(exe_copy, exe_path, sizeof(exe_copy) - 1);
    exe_copy[sizeof(exe_copy) - 1] = '\0';
    char *exe_dir = dirname(exe_copy);

    char resources[4096];
    snprintf(resources, sizeof(resources), "%s/../Resources", exe_dir);

    const char *home = getenv("HOME");
    if (!home || !*home) home = "/tmp";

    // ── 1b. Redirect stdout/err to ~/.flow/app.log (GUI launches only) ──────
    {
        char flow_dir[4096];
        snprintf(flow_dir, sizeof(flow_dir), "%s/.flow", home);
        mkdir(flow_dir, 0755);
        redirect_logs(home);
    }
    fprintf(stderr, "[Flow] === launcher start (pid=%d) ===\n", getpid());
    fflush(stderr);

    // ── 2. First-run install ─────────────────────────────────────────────────
    char install_done[4096];
    snprintf(install_done, sizeof(install_done), "%s/.flow/.install_done", home);

    if (!file_exists(install_done)) {
        int rc = run_installer(resources);
        if (rc != 0 || !file_exists(install_done)) {
            fprintf(stderr, "[Flow] Installation failed (rc=%d)\n", rc);
            return 2;
        }
    }

    // ── 3. dlopen() libpython. Search order:
    //   a) ~/.flow/python/lib/libpython3.X.dylib  (portable Python that
    //      install.sh downloads when no system Python exists)
    //   b) Homebrew  /opt/homebrew/Frameworks/Python.framework/...
    //   c) /usr/local Frameworks (Intel-era Homebrew)
    char portable[6][4096];
    int n_portable = 0;
    {
        const char *minors[] = {"3.12", "3.11", "3.10", NULL};
        for (int i = 0; minors[i] && n_portable < 6; i++) {
            snprintf(portable[n_portable], sizeof(portable[n_portable]),
                     "%s/.flow/python/lib/libpython%s.dylib", home, minors[i]);
            n_portable++;
        }
    }
    const char *system_candidates[] = {
        "/opt/homebrew/Frameworks/Python.framework/Versions/3.12/Python",
        "/opt/homebrew/Frameworks/Python.framework/Versions/3.11/Python",
        "/opt/homebrew/Frameworks/Python.framework/Versions/3.10/Python",
        "/usr/local/Frameworks/Python.framework/Versions/3.12/Python",
        "/usr/local/Frameworks/Python.framework/Versions/3.11/Python",
        "/usr/local/Frameworks/Python.framework/Versions/3.10/Python",
        NULL
    };

    void *lib = NULL;
    const char *loaded_path = NULL;
    const char *py_minor = NULL;

    // (a) Portable Python first (zero system dependencies path)
    for (int i = 0; i < n_portable; i++) {
        lib = dlopen(portable[i], RTLD_NOW | RTLD_GLOBAL);
        if (lib) {
            loaded_path = portable[i];
            if      (strstr(portable[i], "3.12")) py_minor = "3.12";
            else if (strstr(portable[i], "3.11")) py_minor = "3.11";
            else if (strstr(portable[i], "3.10")) py_minor = "3.10";
            else                                   py_minor = "3.12";
            break;
        }
    }
    // (b) System candidates
    if (!lib) {
        for (int i = 0; system_candidates[i]; i++) {
            lib = dlopen(system_candidates[i], RTLD_NOW | RTLD_GLOBAL);
            if (lib) {
                loaded_path = system_candidates[i];
                if      (strstr(system_candidates[i], "/3.12/")) py_minor = "3.12";
                else if (strstr(system_candidates[i], "/3.11/")) py_minor = "3.11";
                else if (strstr(system_candidates[i], "/3.10/")) py_minor = "3.10";
                else                                              py_minor = "3.12";
                break;
            }
        }
    }
    if (!lib) {
        fprintf(stderr, "[Flow] Cannot dlopen Python (portable or system). "
                "Last error: %s\n", dlerror());
        return 3;
    }
    fprintf(stderr, "[Flow] Loaded Python from: %s\n", loaded_path);

    Py_DecodeLocale_fn Py_DecodeLocale = (Py_DecodeLocale_fn)dlsym(lib, "Py_DecodeLocale");
    Py_Main_fn         Py_Main         = (Py_Main_fn)        dlsym(lib, "Py_Main");
    if (!Py_DecodeLocale || !Py_Main) {
        fprintf(stderr, "[Flow] Missing Python symbols in %s\n", loaded_path);
        return 4;
    }

    // ── 4. Tell embedded Python where the venv site-packages live ───────────
    char site_pkgs[4096];
    snprintf(site_pkgs, sizeof(site_pkgs),
             "%s/.flow/venv/lib/python%s/site-packages", home, py_minor);
    setenv("PYTHONPATH", site_pkgs, 1);

    // ── 5. Decide whether to inject flow.py ─────────────────────────────────
    // If we were invoked WITHOUT extra args (the normal case: user double-clicks
    // Flow.app), inject flow.py as argv[1] so Python runs the app.
    //
    // If we were invoked WITH args (e.g. multiprocessing's resource_tracker
    // re-execs sys.executable with `-c "..."`, or pip / -m calls), pass argv
    // through UNCHANGED — otherwise we'd re-launch flow.py inside every worker
    // and end up in an infinite spawn loop.
    int inject_flow = 1;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "-c") == 0 || strcmp(a, "-m") == 0 ||
            (strlen(a) > 3 && strcmp(a + strlen(a) - 3, ".py") == 0)) {
            inject_flow = 0;
            break;
        }
    }

    char flow_py[4096];
    snprintf(flow_py, sizeof(flow_py), "%s/flow.py", resources);

    if (inject_flow && !file_exists(flow_py)) {
        fprintf(stderr, "[Flow] flow.py not found at %s\n", flow_py);
        return 5;
    }

    int new_argc = inject_flow ? argc + 1 : argc;
    wchar_t **wargv = (wchar_t**)calloc((size_t)new_argc + 1, sizeof(wchar_t*));
    wargv[0] = Py_DecodeLocale(argv[0], NULL);  // "Flow"

    if (inject_flow) {
        wargv[1] = Py_DecodeLocale(flow_py, NULL);
        for (int i = 1; i < argc; i++) {
            wargv[i + 1] = Py_DecodeLocale(argv[i], NULL);
        }
    } else {
        for (int i = 1; i < argc; i++) {
            wargv[i] = Py_DecodeLocale(argv[i], NULL);
        }
    }
    wargv[new_argc] = NULL;

    return Py_Main(new_argc, wargv);
}
