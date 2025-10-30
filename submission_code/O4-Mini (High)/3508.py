class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # If k has a 1-bit where n has a 0-bit, we can never set bits, so impossible
        if k & ~n:
            return -1
        # We only need to clear the bits that are 1 in n but 0 in k
        # Count those bits:
        return bin(n & ~k).count('1')