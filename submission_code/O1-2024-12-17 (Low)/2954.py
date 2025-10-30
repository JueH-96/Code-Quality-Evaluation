class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        from collections import defaultdict
        
        freq = defaultdict(int)
        distinct_count = 0
        current_sum = 0
        max_sum = 0
        found = False  # To track if we found at least one valid subarray

        left = 0
        for right in range(len(nums)):
            # Include nums[right] in the window
            current_sum += nums[right]
            freq[nums[right]] += 1
            if freq[nums[right]] == 1:
                distinct_count += 1  # A new distinct element

            # If window size exceeds k, shrink from the left
            if right - left + 1 > k:
                freq[nums[left]] -= 1
                if freq[nums[left]] == 0:
                    distinct_count -= 1
                    del freq[nums[left]]
                current_sum -= nums[left]
                left += 1

            # Check if current window is exactly of size k
            if right - left + 1 == k:
                # Check if distinct elements in the window >= m
                if distinct_count >= m:
                    max_sum = max(max_sum, current_sum)
                    found = True

        return max_sum if found else 0