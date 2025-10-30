class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        n = len(nums)
        # Total number of subarrays
        total_subarrays = n * (n + 1) // 2
        # Median position (1-based); if total_subarrays is even, pick the smaller of the two middles.
        median_pos = (total_subarrays + 1) // 2
        
        # Counts how many subarrays have a distinct count of at most x
        def subarrays_count_at_most_x(x: int) -> int:
            freq = defaultdict(int)
            left = 0
            distinct_count = 0
            count = 0
            for right in range(n):
                freq[nums[right]] += 1
                if freq[nums[right]] == 1:
                    distinct_count += 1
                while distinct_count > x:
                    freq[nums[left]] -= 1
                    if freq[nums[left]] == 0:
                        distinct_count -= 1
                    left += 1
                count += (right - left + 1)
            return count
        
        # Binary search for the smallest x for which subarrays_count_at_most_x(x) >= median_pos
        left, right = 1, len(set(nums))
        while left < right:
            mid = (left + right) // 2
            if subarrays_count_at_most_x(mid) >= median_pos:
                right = mid
            else:
                left = mid + 1
        
        return left