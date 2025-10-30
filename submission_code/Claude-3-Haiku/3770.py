class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n, m = len(str1), len(str2)
        if n > m * 10000:
            return ""
        
        result = []
        for i, c in enumerate(str1):
            if c == 'T':
                result.append(str2)
            else:
                for j in range(m):
                    if str2[j] != str2[j % len(str2)]:
                        result.append(str2[j])
                        break
                else:
                    return ""
        
        return ''.join(result)