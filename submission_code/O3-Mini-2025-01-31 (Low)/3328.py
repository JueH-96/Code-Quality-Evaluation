class Solution:
    def minOperations(self, k: int) -> int:
        # We start with an array [1]. We can increase one element or duplicate an element.
        # A good strategy is to first perform increments on the initial element to raise its
        # value to some x, and then duplicate it until the array's length is n.
        # The sum will then be n*x.
        # We already have one element, so duplications required = n - 1.
        # The cost for increments is (x - 1) (since we start from 1).
        # We need n*x >= k, and we choose the minimal n = ceil(k / x).
        # Therefore, total operations = (x - 1) + (ceil(k / x) - 1).
        # We iterate over possible x (starting value after increments) and choose minimum cost.
        
        ans = float('inf')
        # x takes value from 1 (no increments) to k (worst case, incrementing until you have at least k, and then no duplication needed).
        for x in range(1, k + 1):
            n = (k + x - 1) // x  # This is ceil(k/x)
            ops = (x - 1) + (n - 1)
            if ops < ans:
                ans = ops
            # Since f(x) is not strictly monotonic, we still check all.
        return ans