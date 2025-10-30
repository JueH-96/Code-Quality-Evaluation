from collections import defaultdict
from math import inf
from typing import List

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        # if all elements are equal, answer is 0
        if all(x == nums[0] for x in nums):
            return 0
        
        # map value -> list of indices where it appears
        pos = defaultdict(list)
        for i, num in enumerate(nums):
            pos[num].append(i)
        
        ans = inf
        
        # For each candidate value, compute the maximum gap between its occurrences in circular manner.
        for v, indices in pos.items():
            indices.sort()
            max_gap = 0
            k = len(indices)
            # gap between last and first element over the circular wrap
            wrap_gap = indices[0] + n - indices[-1] - 1
            max_gap = max(max_gap, wrap_gap)
            # gap between consecutive indices
            for i in range(k - 1):
                gap = indices[i+1] - indices[i] - 1
                max_gap = max(max_gap, gap)
            # The value spreads to fill the gap at a rate of 1 index per second from each end.
            # So the number of seconds required is ceil(max_gap / 2).
            # Instead of using math.ceil, we can compute as:
            seconds = (max_gap + 1) // 2
            ans = min(ans, seconds)
            
        return ans

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minimumSeconds([1,2,1,2]))   # Expected output: 1
    print(sol.minimumSeconds([2,1,3,3,2])) # Expected output: 2
    print(sol.minimumSeconds([5,5,5,5]))   # Expected output: 0