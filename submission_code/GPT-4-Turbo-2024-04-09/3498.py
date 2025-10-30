class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        n = len(nums)
        half_n = n // 2
        changes_needed = 0
        
        # Dictionary to store frequency of (nums[i], nums[n-i-1]) pairs
        pair_freq = defaultdict(int)
        
        # Dictionary to store frequency of nums[i] and nums[n-i-1] individually
        left_freq = defaultdict(int)
        right_freq = defaultdict(int)
        
        # Populate the dictionaries
        for i in range(half_n):
            left = nums[i]
            right = nums[n-i-1]
            pair_freq[(left, right)] += 1
            left_freq[left] += 1
            right_freq[right] += 1
        
        # Find the best pair (a, b) such that changing to (a, b) and (b, a) minimizes changes
        max_pair_count = 0
        for (a, b), count in pair_freq.items():
            max_pair_count = max(max_pair_count, count)
        
        # Calculate changes needed by assuming the best pair is used
        changes_needed = n - 2 * max_pair_count
        
        return changes_needed