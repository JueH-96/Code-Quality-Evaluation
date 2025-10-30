class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        count = 0

        # Iterate through all possible ending positions j
        for j in range(n):
            # Get the last digit of the substring ending at j
            last_digit_char = s[j]
            last_digit = int(last_digit_char)

            # Substrings must be divisible by their non-zero last digit
            if last_digit == 0:
                continue

            # For a fixed ending position j and non-zero last digit d = int(s[j]),
            # iterate through all possible starting positions i from j down to 0.
            # This generates all substrings s[i...j] ending at j.
            # We check if int(s[i...j]) is divisible by d.
            # We can calculate int(s[i...j]) % d iteratively by working backwards from j.

            # Let remainder_s_k_to_j be int(s[k...j]) % last_digit.
            # We compute this for k = j, j-1, ..., 0.
            # remainder_s_k_to_j = (int(s[k]) * 10^(j-k) + int(s[k+1...j])) % last_digit
            # Let current_rem_of_suffix be int(s[k+1...j]) % last_digit (0 when k=j)
            # Let power_of_10_for_digit be 10^(j-k) % last_digit (1 when k=j)

            remainder_of_suffix_to_the_right_of_i = 0 # This will store int(s[i+1...j]) % last_digit
            power_of_10_for_digit_at_i = 1 # This will store 10^(j-i) % last_digit

            # Iterate i from j down to 0
            for i in range(j, -1, -1):
                digit = int(s[i])

                # Calculate int(s[i...j]) % last_digit
                # int(s[i...j]) = digit * 10^(j-i) + int(s[i+1...j])
                # Remainder of s[i...j] = (digit * (10^(j-i) % last_digit) + int(s[i+1...j]) % last_digit) % last_digit
                # Using our variables:
                # remainder_s_i_to_j = (digit * power_of_10_for_digit_at_i + remainder_of_suffix_to_the_right_of_i) % last_digit
                
                remainder_s_i_to_j = (digit * power_of_10_for_digit_at_i + remainder_of_suffix_to_the_right_of_i) % last_digit

                if remainder_s_i_to_j == 0:
                    count += 1

                # Prepare for the next iteration (i-1).
                # The suffix to the right of s[i-1] is s[i...j].
                # So, remainder_of_suffix_to_the_right_of_i for step (i-1) is remainder_s_i_to_j.
                remainder_of_suffix_to_the_right_of_i = remainder_s_i_to_j 

                # For the next iteration (i-1), the power of 10 for digit s[i-1] is 10^(j-(i-1)) = 10^(j-i+1).
                # This is 10 times the current power 10^(j-i).
                power_of_10_for_digit_at_i = (power_of_10_for_digit_at_i * 10) % last_digit

        return count