import collections

class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        
        # Calculate the total number of possible substrings in a string of length n.
        # This is n * (n + 1) / 2.
        total_possible_substrings = n * (n + 1) // 2
        
        # We will count "bad" substrings: those where *all* characters appear strictly less than k times.
        # The final result will be total_possible_substrings - bad_substrings_count.
        bad_substrings_count = 0
        
        # Iterate through all possible starting positions (left pointer) of a substring.
        for i in range(n):
            # char_counts: Stores frequencies of characters in the current window s[i:j+1].
            char_counts = collections.Counter()
            
            # num_distinct_in_window: Counts how many distinct characters are in the current window.
            num_distinct_in_window = 0
            
            # num_freq_lt_k: Counts how many distinct characters in the current window
            # have a frequency strictly less than k.
            num_freq_lt_k = 0
            
            # Iterate through all possible ending positions (right pointer) for the current left pointer i.
            for j in range(i, n):
                char = s[j]
                
                # Store the character count before adding the current character s[j].
                char_count_before_add = char_counts[char]
                
                # Add the current character to the window by incrementing its count.
                char_counts[char] += 1
                
                # Get the character count after adding s[j].
                char_count_after_add = char_counts[char]
                
                # Update num_distinct_in_window and num_freq_lt_k based on the character's frequency change.
                if char_count_before_add == 0:
                    # This character is new to the window.
                    num_distinct_in_window += 1
                    # Since its count just became 1, if 1 is less than k, it contributes to num_freq_lt_k.
                    if char_count_after_add < k:
                        num_freq_lt_k += 1
                elif char_count_before_add < k and char_count_after_add >= k:
                    # This character's frequency just crossed the threshold from being < k to >= k.
                    # It no longer contributes to num_freq_lt_k.
                    num_freq_lt_k -= 1
                # In other cases (e.g., both counts < k, or both >= k), num_freq_lt_k doesn't change due to this char.
                
                # Check if the current substring s[i:j+1] is a "bad" substring.
                # A substring is "bad" if and only if all distinct characters within it have a frequency < k.
                # This condition is met when the number of distinct characters in the window
                # is equal to the number of distinct characters whose frequency is less than k.
                if num_distinct_in_window == num_freq_lt_k:
                    bad_substrings_count += 1
                else:
                    # If the current substring s[i:j+1] is NOT bad (meaning at least one character
                    # has a frequency >= k), then any longer substring starting at the same 'i'
                    # (s[i:j'+1] where j' > j) will also contain that character with >= k frequency.
                    # Therefore, all such longer substrings will also NOT be "bad".
                    # We can stop extending the window from 'i' and move to the next starting position.
                    break
                    
        # The total number of desired substrings is total_possible_substrings minus the count of "bad" substrings.
        return total_possible_substrings - bad_substrings_count