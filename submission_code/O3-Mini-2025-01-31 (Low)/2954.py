class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        if n < k:
            return 0
        
        freq = {}
        window_sum = 0
        max_sum = 0
        count_distinct = 0
        
        # Initialize first window
        for i in range(k):
            window_sum += nums[i]
            if nums[i] in freq:
                freq[nums[i]] += 1
            else:
                freq[nums[i]] = 1
                count_distinct += 1
        
        if count_distinct >= m:
            max_sum = window_sum
        
        # Slide the window
        for i in range(k, n):
            # Remove element going out (at index i-k)
            out_elem = nums[i - k]
            freq[out_elem] -= 1
            window_sum -= out_elem
            if freq[out_elem] == 0:
                del freq[out_elem]
                count_distinct -= 1
            
            # Add the new element
            in_elem = nums[i]
            window_sum += in_elem
            if in_elem in freq:
                freq[in_elem] += 1
            else:
                freq[in_elem] = 1
                count_distinct += 1
            
            if count_distinct >= m:
                max_sum = max(max_sum, window_sum)
        
        return max_sum