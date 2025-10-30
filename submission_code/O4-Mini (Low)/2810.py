from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # m[i] will track the minimum price we can pay at position i
        # after performing k shifts, before buying the chocolate at logical "slot" i.
        m = nums[:]  # initially, no shifts done
        total_min = float('inf')
        # Try all possible numbers of shifts k = 0 .. n-1
        # For each k, cost = k*x + sum of best prices at each slot
        for k in range(n):
            curr_cost = k * x + sum(m)
            if curr_cost < total_min:
                total_min = curr_cost
            # Prepare m for the next shift (k+1)
            # After one more shift, the chocolate that was at index i-1 (mod n)
            # moves into slot i, so we can potentially buy that cheaper.
            prev_idx = ( - (k + 1) ) % n
            for i in range(n):
                # nums[(i - (k+1)) mod n] equals nums[(prev_idx + i) mod n]
                candidate = nums[(i + prev_idx) % n]
                if candidate < m[i]:
                    m[i] = candidate
        return total_min

# Example usage:
# sol = Solution()
# print(sol.minCost([20,1,15], 5))  # 13
# print(sol.minCost([1,2,3], 4))    # 6