import math

class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        """
        Finds the shortest beautiful substring with k ones and returns the 
        lexicographically smallest one among those with the minimum length.

        A substring is beautiful if the number of 1's in it is exactly k.

        Args:
            s: The binary string.
            k: The required number of ones in a beautiful substring.

        Returns:
            The lexicographically smallest beautiful substring of the shortest 
            length, or "" if none exists.
        """
        n = len(s)
        # Initialize the minimum length found so far to infinity.
        min_len = math.inf
        # Stores the start index of the best substring found so far.
        # -1 indicates no beautiful substring has been found yet.
        best_start_index = -1 
        
        # Use a sliding window approach with 'left' and 'right' pointers.
        left = 0
        # Keep track of the number of ones in the current window [left, right].
        ones_count = 0

        # Iterate through the string with the 'right' pointer to expand the window.
        for right in range(n):
            # If the character entering the window is '1', increment the count.
            if s[right] == '1':
                ones_count += 1

            # Once the window potentially contains k or more ones, try to shrink it 
            # from the left while maintaining the condition or finding valid substrings.
            # This inner loop ensures we find the shortest valid substring ending at 'right'
            # and handles cases where shrinking might reveal other valid substrings.
            while ones_count >= k: 
                # If the current window [left, right] has exactly k ones, it's beautiful.
                if ones_count == k:
                    current_len = right - left + 1
                    
                    # Check if this beautiful substring is shorter than the minimum length found so far.
                    if current_len < min_len:
                        # Found a new shortest substring. Update min_len and best_start_index.
                        min_len = current_len
                        best_start_index = left 
                    # If it has the same length as the current minimum length.
                    elif current_len == min_len:
                        # We need the lexicographically smallest among those with minimum length.
                        # Compare the current substring with the best one found so far of the same length.
                        
                        # Extract the current candidate substring.
                        # Slicing is s[start : end], where end is exclusive. length = end - start.
                        current_sub = s[left : left + current_len] 
                        # Extract the best substring found so far (guaranteed to exist if min_len is not inf).
                        best_sub_so_far = s[best_start_index : best_start_index + min_len] 

                        # If the current substring is lexicographically smaller, update the best start index.
                        if current_sub < best_sub_so_far:
                            best_start_index = left 
                            
                # Now, shrink the window from the left by moving the 'left' pointer.
                # If the character leaving the window is '1', decrement the count.
                if s[left] == '1':
                    ones_count -= 1
                # Move the left pointer to shrink the window.
                left += 1
                # The while loop condition (ones_count >= k) will be checked again 
                # for the new (shrunk) window [left, right]. This allows us to find
                # the leftmost valid start for the current 'right' endpoint.

        # After iterating through all possible 'right' endpoints.
        if best_start_index == -1:
            # No beautiful substring was found throughout the iteration.
            return ""
        else:
            # A beautiful substring was found. Return the one starting at 
            # best_start_index with length min_len. This is guaranteed to be
            # the shortest and lexicographically smallest one.
            return s[best_start_index : best_start_index + min_len]