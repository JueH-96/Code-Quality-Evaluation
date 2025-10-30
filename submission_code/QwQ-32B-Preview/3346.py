class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        result = []
        for c in s:
            pos_c = ord(c) - ord('a')
            for pos_d in range(26):
                distance = min(abs(pos_d - pos_c), 26 - abs(pos_d - pos_c))
                if distance <= k:
                    result.append(chr(pos_d + ord('a')))
                    k -= distance
                    break
            else:
                # If no character can be changed within k, keep the original character
                result.append(c)
        return ''.join(result)