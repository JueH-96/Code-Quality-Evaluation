from collections import defaultdict

class Solution:
    def medianOfUniquenessArray(self, nums: list) -> int:
        n = len(nums)
        m = n * (n + 1) // 2
        k = (m - 1) // 2
        target = k + 1
        
        # Compute the maximum possible x
        unique = set(nums)
        max_x = len(unique)
        
        def count_subarrays_at_most(x):
            freq = defaultdict(int)
            left = 0
            distinct = 0
            count = 0
            for right in range(n):
                num = nums[right]
                if freq[num] == 0:
                    distinct += 1
                freq[num] += 1
                
                # Ensure the number of distinct elements is <= x
                while distinct > x:
                    left_num = nums[left]
                    freq[left_num] -= 1
                    if freq[left_num] == 0:
                        distinct -= 1
                    left += 1
                
                # Add the number of valid subarrays ending at 'right'
                count += (right - left + 1)
            return count
        
        # Binary search for the minimal x where count >= target
        low = 1
        high = max_x
        while low < high:
            mid = (low + high) // 2
            current = count_subarrays_at_most(mid)
            if current < target:
                low = mid + 1
            else:
                high = mid
        return low