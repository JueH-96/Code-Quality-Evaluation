import collections

class Solution:
    """
    Calculates the number of substrings of a digit string 's' that are divisible
    by their non-zero last digit.
    """
    def countSubstrings(self, s: str) -> int:
        """
        Counts the valid substrings using an O(N) approach based on the properties
        of divisibility by the last digit.

        The approach iterates through the string, considering each index 'j' as the
        potential end of substrings. For each 'j', it calculates the number of
        starting indices 'i' (where i <= j) such that the substring s[i..j]
        satisfies the condition: the last digit s[j] is non-zero, and the integer
        value of s[i..j] is divisible by the integer value of s[j].

        Optimization techniques used:
        - Divisibility by 1, 2, 5: If the last digit is 1, 2, or 5, the divisibility
          condition is always met. We add j+1 to the count.
        - Divisibility by 3, 6, 9: Uses the property that a number is divisible by
          3 (or 9) iff the sum of its digits is divisible by 3 (or 9). This is
          solved efficiently using prefix sums of digits modulo 3 (or 9) and a hash map
          (dictionary) to store counts. For 6, we only need the divisibility by 3
          check since the last digit '6' ensures divisibility by 2.
        - Divisibility by 4, 8: Uses the property that divisibility by 4 (or 8)
          depends only on the last two (or three) digits. This allows for a constant
          time check per ending index 'j'.
        - Divisibility by 7: Uses a mathematical trick involving prefix values modulo 7
          and powers of 10 modulo 7. Let P7[k] = int(s[0...k-1]) % 7 and
          invT7[k] = pow(10, -k, 7). Let Q7[k] = (P7[k] * invT7[k]) % 7. The condition
          int(s[i..j]) % 7 == 0 is equivalent to Q7[j+1] == Q7[i]. This is solved
          similarly to the modulo 3/9 cases using prefix computations and a hash map.

        Args:
            s: The input string of digits.

        Returns:
            The total number of substrings divisible by their non-zero last digit.
        """
        n = len(s)
        ans = 0

        # --- Precomputation ---
        # Arrays to store prefix information (size n+1 for indices 0 to n)

        # P3[k]: sum of digits s[0...k-1] modulo 3
        P3 = [0] * (n + 1)
        # P9[k]: sum of digits s[0...k-1] modulo 9
        P9 = [0] * (n + 1)
        # P7[k]: int(s[0...k-1]) modulo 7
        P7 = [0] * (n + 1)
        # invT7[k]: pow(10, -k, 7) = pow(5, k, 7)
        invT7 = [0] * (n + 1)
        # Q7[k]: (P7[k] * invT7[k]) mod 7
        Q7 = [0] * (n + 1)

        # Calculate P3 and P9 using sum of digits property
        for k in range(n):
            digit_k = int(s[k])
            P3[k+1] = (P3[k] + digit_k) % 3
            P9[k+1] = (P9[k] + digit_k) % 9

        # Calculate P7, invT7, Q7
        invT7[0] = 1
        inv10_mod7 = 5 # Modular inverse of 10 modulo 7 is 5 (10*5 = 50 = 1 mod 7)
        for k in range(n):
            digit_k = int(s[k])
            # P7[k+1] = (int(s[0..k-1]) * 10 + int(s[k])) % 7
            P7[k+1] = (P7[k] * 10 + digit_k) % 7
            # invT7[k+1] = pow(5, k+1, 7) = pow(5, k, 7) * 5 = invT7[k] * 5
            invT7[k+1] = (invT7[k] * inv10_mod7) % 7
            # Q7[k+1] = (P7[k+1] * invT7[k+1]) % 7
            Q7[k+1] = (P7[k+1] * invT7[k+1]) % 7

        # --- Counting using prefix counts ---
        # countsX[val] stores the number of prefix indices k (0 <= k <= j)
        # such that PX[k] = val, where PX represents P3, P9, or Q7.
        counts3 = collections.defaultdict(int)
        counts9 = collections.defaultdict(int)
        counts7 = collections.defaultdict(int)

        # Initialize counts for the empty prefix (index k=0)
        # This corresponds to the start of the string (before index 0)
        counts3[P3[0]] += 1
        counts9[P9[0]] += 1
        counts7[Q7[0]] += 1

        # Iterate through the string, considering each position j as the end of substrings
        for j in range(n):
            digit_j = int(s[j])
            
            # Get the precomputed values for the prefix ending at index j
            # (These correspond to index j+1 in the precomputed arrays)
            current_P3 = P3[j+1]
            current_P9 = P9[j+1]
            current_Q7 = Q7[j+1]

            # Only process if the last digit is non-zero
            if digit_j != 0:
                d = digit_j # The divisor is the last digit

                # Add contributions based on the divisor d
                if d == 1 or d == 2 or d == 5:
                    # Divisibility is always met for these last digits.
                    # Any substring s[i..j] ending in d=1, 2, or 5 works.
                    # Count all possible start indices i (0 to j). There are j+1.
                    ans += (j + 1)
                elif d == 3 or d == 6:
                    # Divisibility by 3 (or 6, as d=6 implies even) requires sum_digits(s[i..j]) % 3 == 0.
                    # This is equivalent to P3[j+1] == P3[i].
                    # We need the count of prefix indices k in [0, j] such that P3[k] == current_P3.
                    # This count is exactly counts3[current_P3] (stored from previous iterations).
                    ans += counts3[current_P3]
                elif d == 4:
                    # Check divisibility by 4 based on the last 1 or 2 digits.
                    count_j = 1 # Substring s[j] ("4") is always divisible by 4.
                    if j > 0 and int(s[j-1:j+1]) % 4 == 0:
                         # If s[j-1]s[j] is divisible by 4, all s[i..j] for i < j are also divisible.
                         # There are j such starting indices (i=0 to j-1).
                         count_j += j
                    ans += count_j
                elif d == 7:
                    # Divisibility by 7 requires val(s[i..j]) % 7 == 0.
                    # This is equivalent to Q7[j+1] == Q7[i].
                    # We need the count of prefix indices k in [0, j] such that Q7[k] == current_Q7.
                    ans += counts7[current_Q7]
                elif d == 8:
                    # Check divisibility by 8 based on the last 1, 2, or 3 digits.
                    count_j = 1 # Substring s[j] ("8") is always divisible by 8.
                    if j > 0 and int(s[j-1:j+1]) % 8 == 0:
                         # Substring s[j-1..j] contributes 1 if divisible by 8.
                         count_j += 1
                    if j > 1 and int(s[j-2:j+1]) % 8 == 0:
                         # If s[j-2]s[j-1]s[j] is divisible by 8, all s[i..j] for i < j-1 are also divisible.
                         # There are j-1 such starting indices (i=0 to j-2).
                         count_j += (j - 1)
                    ans += count_j
                elif d == 9:
                    # Divisibility by 9 requires sum_digits(s[i..j]) % 9 == 0.
                    # This is equivalent to P9[j+1] == P9[i].
                    # We need the count of prefix indices k in [0, j] such that P9[k] == current_P9.
                    ans += counts9[current_P9]

            # Update the counts dictionary for the prefix ending at index j (value P[j+1])
            # This must be done after using the counts for the current j,
            # and it must be done regardless of whether digit_j is 0, as these counts
            # are needed for future calculations.
            counts3[current_P3] += 1
            counts9[current_P9] += 1
            counts7[current_Q7] += 1
            
        return ans