class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        if '0' in num:
            if num[-1] == '0':
                return n - 2
            elif num[-2] == '0':
                return n - 1
            elif '0' in num[:-2]:
                return n - 1 - num[:-2].rindex('0')
            else:
                return n - 1
        elif '5' in num:
            if num[-1] == '5':
                return n - 2
            elif '5' in num[:-2]:
                return n - 1 - num[:-2].rindex('5')
            else:
                return n - 1
        else:
            return n