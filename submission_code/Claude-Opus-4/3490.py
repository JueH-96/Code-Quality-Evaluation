class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Count all even and all odd
        even_count = sum(1 for num in nums if num % 2 == 0)
        odd_count = len(nums) - even_count
        
        # For alternating patterns, we need to track the last element's parity
        # and the length of subsequences ending with even/odd
        alt_even_odd = 0  # starts with even, alternates to odd
        alt_odd_even = 0  # starts with odd, alternates to even
        
        for num in nums:
            if num % 2 == 0:  # current number is even
                # Can extend odd-even pattern (last was odd, now even)
                alt_odd_even = alt_even_odd + 1
            else:  # current number is odd
                # Can extend even-odd pattern (last was even, now odd)
                alt_even_odd = alt_odd_even + 1
        
        # Return the maximum of all four patterns
        return max(even_count, odd_count, alt_even_odd, alt_odd_even)