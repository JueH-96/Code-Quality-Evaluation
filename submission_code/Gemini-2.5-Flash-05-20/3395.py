import collections

class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        
        # Step 1: Calculate character frequencies of s.
        # This provides the total count for each character across all concatenated anagrams.
        s_counts = collections.Counter(s)
        
        # Step 2: Iterate through all possible lengths for string t, from 1 up to n.
        # We start from 1 because we are looking for the MINIMUM possible length.
        # The first length 'current_len_t' that satisfies all conditions will be our answer.
        for current_len_t in range(1, n + 1):
            # Step 3: Condition 1 - The length of t must be a divisor of the length of s.
            # This is because s is formed by concatenating 'k' anagrams of t,
            # so len(s) = k * len(t), which means len(t) must divide len(s).
            if n % current_len_t == 0:
                # Step 4: Calculate k, the number of anagrams of t that form s.
                k = n // current_len_t
                
                # Step 5: Condition 2 - For each character, its count in s must be divisible by k.
                # If count_s(char) is the frequency of 'char' in s, and count_t(char)
                # is the frequency in t, then count_s(char) = k * count_t(char).
                # This implies that count_s(char) must be divisible by k.
                
                is_valid_t_length = True
                # Iterate through the frequencies of characters found in s.
                for char_count_in_s in s_counts.values():
                    if char_count_in_s % k != 0:
                        # If any character's count in s is not perfectly divisible by k,
                        # then this current_len_t is not a possible length for t.
                        is_valid_t_length = False
                        break # No need to check further counts for this current_len_t
                
                # Step 6: If all conditions are met for current_len_t, it's a valid length.
                # Since we iterate current_len_t in increasing order, this is guaranteed
                # to be the minimum possible length.
                if is_valid_t_length:
                    return current_len_t
        
        # This line should theoretically never be reached.
        # The largest possible value for current_len_t is 'n' (when t itself is s).
        # In this case, k = n // n = 1. Any character count `char_count_in_s`
        # is always perfectly divisible by 1 (`char_count_in_s % 1 == 0`).
        # Therefore, 'n' will always be a valid length, ensuring a return value
        # within the loop.
        return n