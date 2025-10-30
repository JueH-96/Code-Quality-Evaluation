class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        if n == 1:
            return derived[0] == 0
        xor_sum = 0
        for i in range(n - 1):
            xor_sum ^= derived[i]
        return xor_sum == derived[-1]