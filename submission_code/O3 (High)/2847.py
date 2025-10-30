from typing import List

class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        """
        We need to count how many disjoint pairs (i, j) we can form such that
        words[i] == reversed(words[j]) with i < j. 
        All words are distinct and of length 2.

        Because each word can appear at most once in the list and therefore
        at most once in a pair, every valid pair is simply a word together with
        its reverse (which must also appear exactly once in the list).
        Hence the maximum number of pairs equals the count of such reversible
        matches.
        """
        seen = set()
        pairs = 0

        for w in words:
            rev = w[::-1]
            if rev in seen:
                # The reverse has already been encountered and is unused,
                # so we can form a pair.
                pairs += 1
                seen.remove(rev)  # remove to mark its usage (optional)
            else:
                seen.add(w)

        return pairs