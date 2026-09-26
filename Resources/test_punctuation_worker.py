import unittest
from punctuation_worker import apply_labels, protected

class PunctuationTests(unittest.TestCase):
    def test_exact_spacing(self):
        self.assertEqual(apply_labels("ciao  mondo\ncome stai", [(".",.9),("0",1),("0",1),("?",.9)]), "ciao.  mondo\ncome stai?")
    def test_protected(self):
        for word in ["123,45", "AB-2026", "x_y", "test@example.com", "https://example.com", "main.swift", "/tmp/a", "x=1"]:
            self.assertTrue(protected(word))
            self.assertEqual(apply_labels(word, [(".",.99)]), word)
    def test_no_duplicate_punctuation(self):
        for word in ["ciao!", "ciao.", "ciao,", "ciao?", "ciao:"]:
            self.assertEqual(apply_labels(word, [(".",.99)]), word)
    def test_low_confidence_and_unknown_labels(self):
        for label, confidence in [(".",.74),("!",.99),("0",1),("INJECT",1),("-",.99)]:
            self.assertEqual(apply_labels("ciao", [(label,confidence)]), "ciao")
    def test_mismatch(self):
        self.assertEqual(apply_labels("ciao mondo", [(".",1)]), "ciao mondo")

if __name__ == "__main__": unittest.main()
