class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        def at_most_k_distinct(k):
            count = defaultdict(int)
            left = 0
            result = 0
            for right in range(len(nums)):
                if count[nums[right]] == 0:
                    k -= 1
                count[nums[right]] += 1
                
                while k < 0:
                    count[nums[left]] -= 1
                    if count[nums[left]] == 0:
                        k += 1
                    left += 1
                
                result += right - left + 1
            return result
        
        total_distinct = len(set(nums))
        return at_most_k_distinct(total_distinct) - at_most_k_distinct(total_distinct - 1)