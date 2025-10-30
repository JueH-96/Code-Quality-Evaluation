class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        remaining_k = k
        res = []
        for char in s:
            for candidate in 'abcdefghijklmnopqrstuvwxyz':
                # compute the distance between current char and candidate
                diff = abs(ord(char) - ord(candidate))
                distance = min(diff, 26 - diff)
                if distance <= remaining_k:
                    res.append(candidate)
                    remaining_k -= distance
                    break  # move to the next character
        return ''.join(res)