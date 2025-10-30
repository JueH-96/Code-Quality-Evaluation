from collections import defaultdict

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        max_sum = float('-inf')
        current_sum = 0
        start = 0
        end = 0
        num_dict = defaultdict(list)
        
        while end < len(nums):
            current_sum += nums[end]
            num_dict[nums[end]].append(end)
            
            while abs(nums[start] - nums[end]) != k:
                current_sum -= nums[start]
                num_dict[nums[start]].pop(0)
                if not num_dict[nums[start]]:
                    del num_dict[nums[start]]
                start += 1
                
            max_sum = max(max_sum, current_sum)
            end += 1
            
        return max_sum if max_sum != float('-inf') else 0