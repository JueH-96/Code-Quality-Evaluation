import itertools

class Solution:
    def validStrings(self, n: int) -> List[str]:
        result = []
        for bits in itertools.product('01', repeat=n):
            s = ''.join(bits)
            valid = True
            for i in range(len(s) - 1):
                if s[i] == '0' and s[i+1] == '0':
                    valid = False
                    break
            if valid:
                result.append(s)
        return result