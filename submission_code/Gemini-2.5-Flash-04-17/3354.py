class Solution:
    def minimizeStringValue(self, s: str) -> str:
        n = len(s)
        
        # Step 1: Determine the counts of each character to be used for '?'
        
        # Count initial occurrences of each character and '?'
        initial_counts = [0] * 26
        q_count = 0
        for char in s:
            if char == '?':
                q_count += 1
            else:
                initial_counts[ord(char) - ord('a')] += 1

        # temp_counts will be modified to find which character gets the next '?'
        # This is conceptually the counts including the '?'s assigned so far
        temp_counts = list(initial_counts)
        
        # replacement_counts stores how many times each char 'a'-'z' will replace a '?'
        replacement_counts = [0] * 26
        
        # Assign Q question marks to characters based on current minimum count
        # using lexicographical order as tie-breaker.
        for _ in range(q_count):
            min_count = float('inf')
            min_char_idx = -1
            
            # Find the character with the minimum current count
            # Iterate from 'a' to 'z' to ensure lexicographical tie-breaking
            for i in range(26):
                if temp_counts[i] < min_count:
                    min_count = temp_counts[i]
                    min_char_idx = i
            
            # Increment the count for this character in replacement_counts
            replacement_counts[min_char_idx] += 1
            # Conceptually increment the count in temp_counts for future assignments
            temp_counts[min_char_idx] += 1

        # Step 2: Construct the final string
        
        # Convert the string to a mutable list of characters
        s_list = list(s)
        
        # Pointer for replacement_counts array
        char_idx = 0 

        # Iterate through the string and replace '?'
        for i in range(n):
            if s_list[i] == '?':
                # Find the smallest character index that still has replacements available
                # Since replacement_counts is populated based on minimum counts
                # with lexicographical tie-breaking, iterating char_idx
                # from 0 upwards guarantees we use the smallest available character
                # for the current '?' position (from left to right).
                while char_idx < 26 and replacement_counts[char_idx] == 0:
                    char_idx += 1
                
                # Replace the '?' with the character corresponding to char_idx
                s_list[i] = chr(ord('a') + char_idx)
                # Decrement the count as one replacement is used
                replacement_counts[char_idx] -= 1

        # Join the list back into a string
        return "".join(s_list)