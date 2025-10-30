class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        # Option 1: Make all '0's
        cost0 = 0
        current_flip = 0
        for i in range(n):
            bit = int(s[i])
            if current_flip:
                bit ^= 1
            if bit != 0:
                # Flip from i to n-1
                cost0 += n - i
                current_flip ^= 1
        # Option 2: Make all '1's
        cost1 = 0
        current_flip = 0
        for i in range(n):
            bit = int(s[i])
            if current_flip:
                bit ^= 1
            if bit != 1:
                # Flip from i to n-1
                cost1 += n - i
                current_flip ^= 1
        return min(cost0, cost1)