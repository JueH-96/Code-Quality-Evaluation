class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        # We want subarrays that contain the global maximum (let's call it M) at least k times.
        M = max(nums)
        
        # Count how many times M appears in nums.
        total_count_of_M = sum(1 for x in nums if x == M)
        
        # If M appears fewer than k times in the entire array, the answer is 0.
        if total_count_of_M < k:
            return 0
        
        # Helper function to count how many subarrays have at most "K" occurrences of M.
        def at_most_K_M(K: int) -> int:
            count_M = 0       # how many M's in the current window
            left = 0
            res = 0
            for right, val in enumerate(nums):
                if val == M:
                    count_M += 1
                while count_M > K:
                    if nums[left] == M:
                        count_M -= 1
                    left += 1
                # All subarrays ending at right with valid count
                res += (right - left + 1)
            return res
        
        # Number of subarrays with at least k M's =
        #   count of subarrays with at most total_count_of_M M's
        #   minus count of subarrays with at most (k-1) M's
        return at_most_K_M(total_count_of_M) - at_most_K_M(k - 1)