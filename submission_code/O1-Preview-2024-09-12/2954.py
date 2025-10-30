class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        from collections import defaultdict
        
        n = len(nums)
        freq = defaultdict(int)
        curr_sum = 0
        max_sum = 0

        # Initialize the first window
        for i in range(k):
            curr_sum += nums[i]
            freq[nums[i]] += 1

        if len(freq) >= m:
            max_sum = curr_sum

        # Slide the window
        for i in range(k, n):
            # Remove the element going out of the window
            out_elem = nums[i - k]
            freq[out_elem] -=1
            if freq[out_elem] == 0:
                del freq[out_elem]
            curr_sum -= out_elem

            # Add the new element
            in_elem = nums[i]
            freq[in_elem] += 1
            curr_sum += in_elem

            if len(freq) >= m:
                max_sum = max(max_sum, curr_sum)

        return max_sum