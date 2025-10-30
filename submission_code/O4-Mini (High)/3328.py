class Solution:
    def minOperations(self, k: int) -> int:
        # We look for a final element value v >= 1, do (v-1) increments on the initial 1,
        # then duplicate that v-valued element c-1 times, where c = ceil(k/v).
        # Total operations = (v - 1) + (c - 1) = v + c - 2.
        # Minimize over v.
        
        ans = float('inf')
        for v in range(1, k + 1):
            # Number of elements needed to reach sum >= k when each has value v
            c = (k + v - 1) // v
            ops = v + c - 2
            if ops < ans:
                ans = ops
        return ans