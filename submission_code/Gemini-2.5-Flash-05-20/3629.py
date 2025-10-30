import collections

class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7

        # counts[i] will store the number of occurrences of the character chr(ord('a') + i)
        # in the current string.
        # Initialize with zeros for all 26 lowercase English letters.
        counts = [0] * 26

        # Populate initial counts from the input string s.
        # For each character in 's', increment its corresponding count.
        for char_val in s:
            counts[ord(char_val) - ord('a')] += 1

        # Perform t transformations.
        for _ in range(t):
            # next_counts will store the character counts after the current transformation.
            # It's initialized to zeros at the beginning of each new transformation step.
            next_counts = [0] * 26

            # Rule 1: Handle characters 'a' through 'y' (indices 0 to 24).
            # If a character is 'char(i)' (where i is 0 to 24), it transforms into 'char(i+1)'.
            # Therefore, the count of 'char(i)' (counts[i]) contributes to the count of
            # 'char(i+1)' (next_counts[i+1]).
            for i in range(25): # Loop for 'a' (index 0) through 'y' (index 24)
                next_counts[i+1] = (next_counts[i+1] + counts[i]) % MOD
            
            # Rule 2: Handle character 'z' (index 25).
            # If a character is 'z', it transforms into the string "ab".
            # This means the count of 'z' (counts[25]) contributes to both:
            # - The count of 'a' (next_counts[0])
            # - The count of 'b' (next_counts[1])
            next_counts[0] = (next_counts[0] + counts[25]) % MOD
            next_counts[1] = (next_counts[1] + counts[25]) % MOD
            
            # After processing all transformations for the current step,
            # update the current counts array to the new counts for the next iteration.
            counts = next_counts

        # The total length of the string after 't' transformations is simply the sum of
        # all character counts in the final 'counts' array. Each count represents a character
        # that contributes 1 to the total length.
        # Since individual counts are already modulo MOD, their sum might still exceed MOD,
        # so we take modulo MOD one last time for the final result.
        final_length = sum(counts) % MOD

        return final_length