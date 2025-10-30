class Solution:
    def doesValidArrayExist(self, derived):
        n = len(derived)
        if n == 1:
            return True
        original = [0] * n
        original[0] = derived[0] ^ derived[1]
        for i in range(1, n):
            original[i] = derived[i] ^ original[i - 1]
        return original[0] == (original[-1] ^ derived[0])