class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        # 1) Identify the global maximum M
        M = max(nums)
        
        # 2) Count how many times M appears in nums
        count_M = sum(1 for x in nums if x == M)
        
        # If M appears fewer than k times overall, no subarray can have M >= k times
        if count_M < k:
            return 0
        
        # A helper function that returns how many subarrays have at most x occurrences of M
        def atMostXOccurrences(x: int) -> int:
            left = 0
            freq = 0  # how many times M appears in the current window
            result = 0
            
            for right, val in enumerate(nums):
                if val == M:
                    freq += 1
                
                # Shrink from the left while we have more than x occurrences in the window
                while freq > x:
                    if nums[left] == M:
                        freq -= 1
                    left += 1
                
                # All subarrays ending at 'right' with freq <= x
                result += (right - left + 1)
            
            return result
        
        # Total number of subarrays in an array of length n
        n = len(nums)
        total_subarrays = n * (n + 1) // 2
        
        # Number of subarrays with at least k occurrences of M
        # = total_subarrays - number of subarrays with at most (k-1) occurrences of M
        return total_subarrays - atMostXOccurrences(k - 1)