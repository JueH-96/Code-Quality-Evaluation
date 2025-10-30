from collections import defaultdict

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        total_subarrays = n * (n + 1) // 2
        median_pos = (total_subarrays + 1) // 2
        
        def count_subarrays_with_at_most_k_distinct(k):
            count = 0
            left = 0
            freq = defaultdict(int)
            distinct = 0
            for right in range(n):
                if freq[nums[right]] == 0:
                    distinct += 1
                freq[nums[right]] += 1
                while distinct > k:
                    freq[nums[left]] -= 1
                    if freq[nums[left]] == 0:
                        distinct -= 1
                    left += 1
                count += right - left + 1
            return count
        
        low, high = 1, len(set(nums))
        while low < high:
            mid = (low + high) // 2
            if count_subarrays_with_at_most_k_distinct(mid) >= median_pos:
                high = mid
            else:
                low = mid + 1
        return low