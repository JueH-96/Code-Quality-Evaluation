import collections

class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        total_count = 0

        # For d=7, we use a DP approach.
        # dp7[r] will store the count of substrings ending at the *previous* index
        # with a value of r modulo 7.
        dp7 = [0] * 7

        # For d=3, 6, 9, we use a prefix sum of digits approach.
        # sum_mod_3/9 stores the running sum of digits modulo 3/9 up to the *previous* index.
        # counts_mod_3/9 stores the frequency of prefix sum remainders encountered so far.
        # We initialize with a count of 1 for a remainder of 0, representing the empty prefix.
        sum_mod_3 = 0
        counts_mod_3 = collections.defaultdict(int)
        counts_mod_3[0] = 1
        
        sum_mod_9 = 0
        counts_mod_9 = collections.defaultdict(int)
        counts_mod_9[0] = 1

        for i in range(n):
            d_val = int(s[i])

            # --- Part 1: Update states for the current index i ---

            # Update dp7 for substrings ending at i.
            # new_dp7 will store counts for substrings ending at the *current* index i.
            new_dp7 = [0] * 7
            new_dp7[d_val % 7] = 1  # For the single-digit substring s[i]
            for r in range(7):
                if dp7[r] > 0:
                    new_rem = (r * 10 + d_val) % 7
                    new_dp7[new_rem] += dp7[r]

            # Update prefix sums to include the current digit.
            current_sum_mod_3 = (sum_mod_3 + d_val) % 3
            current_sum_mod_9 = (sum_mod_9 + d_val) % 9

            # --- Part 2: Add to total_count based on d_val ---
            
            # The last digit must be non-zero for divisibility.
            if d_val != 0:
                if d_val in {1, 2, 5}:
                    # Any integer is divisible by 1.
                    # An integer is divisible by 2 if its last digit is even (d_val=2 is even).
                    # An integer is divisible by 5 if its last digit is 0 or 5 (d_val=5).
                    # In all these cases, any substring ending with d_val is valid.
                    # There are i+1 such substrings (from s[0..i] to s[i..i]).
                    total_count += i + 1
                elif d_val in {3, 6}:
                    # An integer is divisible by 3 if the sum of its digits is divisible by 3.
                    # An integer is divisible by 6 if it's divisible by 2 (true, ends in 6) and 3.
                    # We need to count j (0<=j<=i) where sum_digits(s[j..i]) % 3 == 0.
                    # This means (S[i] - S[j-1]) % 3 == 0 => S[i] % 3 == S[j-1] % 3.
                    # The number of such j's is the count of prefixes (including the empty one)
                    # ending before i, whose sum of digits has the same remainder as S[i].
                    total_count += counts_mod_3[current_sum_mod_3]
                elif d_val == 4:
                    # Divisibility by 4 depends on the last two digits.
                    # The substring s[i] ("4") is always valid.
                    count = 1
                    if i > 0:
                        # For j < i, int(s[j..i]) % 4 == int(s[i-1..i]) % 4.
                        if int(s[i-1:i+1]) % 4 == 0:
                            # All i substrings starting from j=0 to j=i-1 are valid.
                            count += i
                    total_count += count
                elif d_val == 7:
                    # new_dp7[0] is the count of substrings ending at index i divisible by 7.
                    total_count += new_dp7[0]
                elif d_val == 8:
                    # Divisibility by 8 depends on the last three digits.
                    count = 0
                    # Length 1 substring s[i] ("8")
                    if d_val % 8 == 0:  # Always true for d_val=8
                        count += 1
                    # Length 2 substring s[i-1..i]
                    if i > 0 and int(s[i-1:i+1]) % 8 == 0:
                        count += 1
                    # Substrings of length 3 or more
                    # For j <= i-2, int(s[j..i]) % 8 == int(s[i-2..i]) % 8.
                    if i > 1 and int(s[i-2:i+1]) % 8 == 0:
                        # All i-1 substrings from s[0..i] to s[i-2..i] are valid.
                        count += (i - 1)
                    total_count += count
                elif d_val == 9:
                    # Similar to d=3, but with modulo 9.
                    total_count += counts_mod_9[current_sum_mod_9]

            # --- Part 3: Update states for the next iteration ---
            dp7 = new_dp7
            sum_mod_3 = current_sum_mod_3
            counts_mod_3[sum_mod_3] += 1
            sum_mod_9 = current_sum_mod_9
            counts_mod_9[sum_mod_9] += 1
            
        return total_count