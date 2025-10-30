class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        result = []
        for char in s:
            if k == 0:
                result.append(char)
            else:
                diff = min((ord(char) - ord('a') + k) % 26, (ord('z') - ord(char) + 1 + k) % 26)
                if diff <= k:
                    result.append(chr((ord(char) - ord('a') + diff) % 26 + ord('a')))
                    k -= diff
                else:
                    result.append('a')
        return ''.join(result)