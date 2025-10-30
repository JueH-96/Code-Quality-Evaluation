class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        if k == 0:
            return 1
        
        n = len(nums)
        min_length = float('inf')
        left = 0
        
        # Count of how many numbers have each bit set
        bit_count = [0] * 32
        
        def get_or_value():
            result = 0
            for i in range(32):
                if bit_count[i] > 0:
                    result |= (1 << i)
            return result
        
        def add_number(num):
            for i in range(32):
                if num & (1 << i):
                    bit_count[i] += 1
        
        def remove_number(num):
            for i in range(32):
                if num & (1 << i):
                    bit_count[i] -= 1
        
        # Sliding window
        for right in range(n):
            add_number(nums[right])
            
            # Try to shrink window from left
            while left <= right and get_or_value() >= k:
                min_length = min(min_length, right - left + 1)
                remove_number(nums[left])
                left += 1
        
        return min_length if min_length != float('inf') else -1