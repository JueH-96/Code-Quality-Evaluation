class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Count the number of even and odd elements
        even_count = 0
        odd_count = 0
        
        # For alternating subsequence
        max_alt_ending_even = 0  # max length of alternating subsequence ending with even
        max_alt_ending_odd = 0   # max length of alternating subsequence ending with odd
        
        for num in nums:
            if num % 2 == 0:  # even
                even_count += 1
                # Add this even number to a subsequence ending with odd
                max_alt_ending_even = max(max_alt_ending_even, max_alt_ending_odd + 1)
            else:  # odd
                odd_count += 1
                # Add this odd number to a subsequence ending with even
                max_alt_ending_odd = max(max_alt_ending_odd, max_alt_ending_even + 1)
        
        # For subsequences with all elements having the same parity
        all_even_length = even_count if even_count >= 2 else 0
        all_odd_length = odd_count if odd_count >= 2 else 0
        
        # For alternating subsequence
        alt_length = max(max_alt_ending_even, max_alt_ending_odd)
        alt_length = alt_length if alt_length >= 2 else 0
        
        return max(all_even_length, all_odd_length, alt_length)