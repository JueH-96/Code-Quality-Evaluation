class Solution:
    def calculateScore(self, s: str) -> int:
        score = 0
        from collections import defaultdict
        last_occurrence = defaultdict(list)  # char -> list of indices
        
        for i, c in enumerate(s):
            mirror_char = chr(219 - ord(c))
            if last_occurrence[mirror_char]:
                j = last_occurrence[mirror_char].pop()
                score += i - j
            else:
                last_occurrence[c].append(i)
        
        return score