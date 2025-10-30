class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = 0
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                cost = 0
                temp_subarray = subarray[:]
                
                is_non_decreasing = True
                for l in range(1, len(temp_subarray)):
                    if temp_subarray[l] < temp_subarray[l-1]:
                        diff = temp_subarray[l-1] - temp_subarray[l]
                        cost += diff
                        
                        
                if cost <= k:
                    count += 1
        return count