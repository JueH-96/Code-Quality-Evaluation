class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        res = []
        remaining_k = k
        candidates = 'abcdefghijklmnopqrstuvwxyz'
        for char in s:
            original_pos = ord(char) - ord('a')
            for candidate in candidates:
                candidate_pos = ord(candidate) - ord('a')
                diff = abs(original_pos - candidate_pos)
                distance = min(diff, 26 - diff)
                if distance <= remaining_k:
                    res.append(candidate)
                    remaining_k -= distance
                    break
        return ''.join(res)