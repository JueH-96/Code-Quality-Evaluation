class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        # Predefine coprime sets for last digit 1-9
        coprime_with_last = [
            [],  # 0, not used
            (1, 2, 3, 4, 5, 6, 7, 8, 9),  # for last digit 1
            (1, 3, 5, 7, 9),             # for last digit 2
            (1, 2, 4, 5, 7, 8),          # for last digit 3
            (1, 3, 5, 7, 9),             # for last digit 4
            (1, 2, 3, 4, 6, 7, 8, 9),    # for last digit 5
            (1, 5, 7),                   # for last digit 6
            (1, 2, 3, 4, 5, 6, 8, 9),    # for last digit 7
            (1, 3, 5, 7, 9),             # for last digit 8
            (1, 2, 4, 5, 7, 8)           # for last digit 9
        ]
        
        def get_first_digit(num):
            while num >= 10:
                num //= 10
            return num
        
        freq = [0] * 10  # Frequency of first digits, indices 1-9 used
        count = 0
        
        for num in nums:
            first_digit = get_first_digit(num)
            last_digit = num % 10
            # Get the coprime digits for the current last digit
            coprime_digits = coprime_with_last[last_digit]
            # Sum the frequency of coprime first digits seen so far
            for d in coprime_digits:
                count += freq[d]
            # Add the current number's first digit to the frequency
            freq[first_digit] += 1
        
        return count