# FlowSwift reconstruction

Work in progress, not installed. The installed app is preserved.

Sources were reconstructed from local Claude Write/Read/Edit records and the original Codex IPC patch. Mixed historical snapshots require compilation and behavioral verification; this is not a byte-exact recovery of the latest source.

Reconstructed manually: executable entry points; damaged overlapping HistoryDB snippet/stat sections; language recognition via NaturalLanguage and audio conversion via FluidAudio. Transcript filters are intentionally conservative replacements, not recovered originals: explicit non-speech markers removed, legitimate thanks/goodbyes and ambiguous repeated sentences retained. Original backups and recovery scripts are preserved in the parent directory.

The old Qwen implementation and its obsolete prompt-cache tests are archived under Historical. They are not built. Active backend uses the offline FullStop base token classifier (last-subtoken alignment, punctuation insertion only, conservative 0.75 threshold). It does not rewrite grammar, tone, or meaning. Tones are removed from visible settings. Unsupported detected languages and overlong input pass through unchanged. A long-lived Python worker loads only local safetensors; failures/timeouts preserve text. Runtime: ~/Desktop/LLM-Lab/engines/hybrid-s1/bin/python; weights: ~/Desktop/LLM-Lab/models/flow-fullstop-base.

Verified September 21: release app and CLI build, 19 Swift tests, 5 Python unit tests; synthetic Italian audio correctly transcribed; staged app IPC health reached idle/modelReady=true with input hooks disabled and isolated data/socket. Exploratory 19 text cases preserve characters except punctuation insertions, including numbers, URLs and code tokens. These are smoke tests, not a holdout or proof of general quality. The model missed a question mark in “ciao come stai”; first load is roughly 5 seconds, hot inference around 20 ms in this small sample. Warmup starts on Fn press. Normal microphone and foreground-paste acceptance still requires a real user dictation.

Acceptance: build app/CLI; run recovered and new tests; verify Parakeet with synthetic audio; verify non-pasting IPC; select offline cleanup using Italian synthetic fixtures preserving numbers, names, negations and content; stage signed app and retain rollback before installation. Local Bonsai evaluation is a separate follow-on, not evidence of Flow correctness.
