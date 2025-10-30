import collections
import heapq

class Solution:
    def minimizeStringValue(self, s: str) -> str:
        n = len(s)
        
        # Step 1: Count initial frequencies of fixed characters and total '?' count.
        # We'll use a list of 26 integers for counts for efficiency.
        # Index 0 for 'a', 1 for 'b', ..., 25 for 'z'.
        initial_char_counts = [0] * 26
        q_count = 0
        
        for char_s in s:
            if char_s == '?':
                q_count += 1
            else:
                initial_char_counts[ord(char_s) - ord('a')] += 1
        
        # Step 2: Determine which characters should replace '?'s to minimize the total value.
        # The total value is sum_{char c} (count(c) * (count(c) - 1) / 2).
        # To minimize this, we want to make the final counts of all characters as balanced as possible.
        # We achieve this by assigning the `q_count` '?'s to characters that currently have the lowest frequency.
        
        # Use a min-heap to keep track of characters by their current frequency.
        # Each item in the heap will be a tuple: (current_frequency, character_numeric_value).
        # The character_numeric_value (0 for 'a', 1 for 'b', etc.) helps break ties lexicographically
        # when frequencies are equal, as heapq pops the smallest item first.
        min_heap = []
        for char_code in range(26):
            # Push (current_frequency, character_numeric_value) to the heap.
            min_heap.append((initial_char_counts[char_code], char_code))
        
        heapq.heapify(min_heap) # Build the min-heap
        
        # Store the characters chosen for '?'s.
        assigned_q_chars = [] 
        
        for _ in range(q_count):
            # Pop the character with the minimum current count (and smallest char_code for ties).
            current_freq, char_code = heapq.heappop(min_heap)
            
            # This character is chosen for one '?'
            assigned_q_chars.append(chr(ord('a') + char_code))
            
            # Increment its count and push it back to the heap.
            # Its new frequency will be current_freq + 1.
            heapq.heappush(min_heap, (current_freq + 1, char_code))
            
        # Step 3: Sort the chosen characters for '?'s to ensure lexicographical smallest output.
        # This is because '?'s at earlier positions should be replaced by smaller characters
        # from the set of available '?' replacements.
        assigned_q_chars.sort()
        
        # Step 4: Construct the final result string.
        # Iterate through the original string `s`.
        # If `s[i]` is a fixed letter, use it directly.
        # If `s[i]` is '?', take the next character from the sorted `assigned_q_chars`.
        
        result_list = []
        q_idx = 0 # Pointer for assigned_q_chars
        
        for char_s in s:
            if char_s == '?':
                result_list.append(assigned_q_chars[q_idx])
                q_idx += 1
            else:
                result_list.append(char_s)
                
        return "".join(result_list)