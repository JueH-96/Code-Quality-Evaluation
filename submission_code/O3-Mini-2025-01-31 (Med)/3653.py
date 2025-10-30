from typing import List
import math

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        # We use prefix sums.
        # Let prefix[i] be the sum of nums[0]...nums[i-1]. Then the sum of subarray [i, j-1] is prefix[j] - prefix[i].
        # We want (j - i) % k == 0, which is equivalent to j % k == i % k.
        # For each remainder group r, we will maintain the minimum prefix sum encountered so far (for an index with that remainder).
        
        best = -math.inf
        prefix = 0
        # dictionary to store the minimum prefix sum for a given remainder (i.e. index mod k)
        rem_min = {}
        # start with prefix sum 0 at index 0, remainder 0
        rem_min[0] = 0
        
        for j, num in enumerate(nums, 1):
            prefix += num
            r = j % k
            # if a prefix with same remainder exists, update best answer
            if r in rem_min:
                candidate = prefix - rem_min[r]
                if candidate > best:
                    best = candidate
            # update the minimal prefix sum for current remainder
            if r in rem_min:
                rem_min[r] = min(rem_min[r], prefix)
            else:
                rem_min[r] = prefix
                
        return best

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maxSubarraySum([1, 2], 1))          # Expected output: 3
    print(sol.maxSubarraySum([-1,-2,-3,-4,-5], 4)) # Expected output: -10
    print(sol.maxSubarraySum([-5,1,2,-3,4], 2))    # Expected output: 4