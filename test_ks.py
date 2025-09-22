import os
import sys
import subprocess
import unittest

from ks import compute_best_segmentation_score


class TestKsTokenizer(unittest.TestCase):
    def _run_cli(self, stdin_text: str) -> str:
        ks_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ks.py")
        result = subprocess.run(
            [sys.executable, ks_path],
            input=stdin_text,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()

    def test_sample1_cli(self):
        # From prompt: expected 0 because cannot fully segment ("pie" missing)
        stdin_text = """applepie
2
pen 3
apple 10
2
pen apple 5
pie apple 2
"""
        out = self._run_cli(stdin_text)
        self.assertEqual(out, "0")

    def test_sample2_cli(self):
        # From prompt: expected 26
        stdin_text = """goodeats
4
good 15
goo 12
deats 14
eats 10
1
good eats -5
"""
        out = self._run_cli(stdin_text)
        self.assertEqual(out, "26")

    def test_empty_text(self):
        text = ""
        vocab = {"a": 1}
        transitions = {}
        self.assertEqual(compute_best_segmentation_score(text, vocab, transitions), 0)

    def test_no_vocab(self):
        text = "abc"
        vocab = {}
        transitions = {}
        self.assertEqual(compute_best_segmentation_score(text, vocab, transitions), 0)

    def test_unsegmentable_returns_zero(self):
        text = "axc"
        vocab = {"a": 1, "c": 2}
        transitions = {}
        self.assertEqual(compute_best_segmentation_score(text, vocab, transitions), 0)

    def test_negative_transition_affects_choice(self):
        # Construct case where best uses transition bonus across multi-length tokens
        # text: "abc"
        # Options:
        #  - "a"+"bc": 1 + 1 + bonus(a,bc)=3 => 5 (best)
        #  - "ab"+"c": 1 + 1 + bonus(ab,c)=0 => 2
        #  - "a"+"b"+"c": 1+1+1 + bonus(a,b)=-1 + bonus(b,c)=0 => 2
        text = "abc"
        vocab = {"a": 1, "b": 1, "c": 1, "ab": 1, "bc": 1}
        transitions = {("a", "b"): -1, ("ab", "c"): 0, ("a", "bc"): 3}
        self.assertEqual(compute_best_segmentation_score(text, vocab, transitions), 5)


if __name__ == "__main__":
    unittest.main()

