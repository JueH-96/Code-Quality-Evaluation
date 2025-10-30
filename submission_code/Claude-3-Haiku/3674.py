class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            min_val = nums[i]
            for j in range(i, n):
                min_val = min(min_val, nums[j])
                if j > i and nums[j] < min_val:
                    if k >= j - i:
                        k -= j - i
                        count += 1
                        break
                    else:
                        break
                elif j == n - 1:
                    count += 1
        
        return count