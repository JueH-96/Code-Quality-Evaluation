class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        from collections import defaultdict

        freq = defaultdict(int)
        left = 0
        max_len = 0
        count_exceed_k = 0  # tracks how many distinct elements currently exceed freq k
        
        for right in range(len(nums)):
            freq[nums[right]] += 1
            # if freq of this element just became k+1, increment count_exceed_k
            if freq[nums[right]] == k + 1:
                count_exceed_k += 1
            
            # shrink the window if any element's frequency is above k
            while count_exceed_k > 0:
                freq[nums[left]] -= 1
                # if freq of nums[left] just became k, decrement count_exceed_k
                if freq[nums[left]] == k:
                    count_exceed_k -= 1
                left += 1
            
            # update max_len for any valid window
            max_len = max(max_len, right - left + 1)
        
        return max_len