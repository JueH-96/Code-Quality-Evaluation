class Solution:
    def maxOperations(self, s: str) -> int:
        zeros = 0
        operations = 0
        for i in range(len(s) - 1, -1, -1):
            if s[i] == '0':
                zeros += 1
            else:
                if zeros > 0:
                    operations += 1
                    zeros -= 1
        return operations