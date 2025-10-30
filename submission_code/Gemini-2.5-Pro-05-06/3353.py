class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        n = len(s)
        
        # If the string length is less than 2, no substring of length 2 can exist.
        if n < 2:
            return False
        
        # Create the reversed version of s.
        # Example: s = "leetcode", reversed_s = "edocteel"
        # This takes O(n) time and O(n) space.
        reversed_s = s[::-1]
        
        # Iterate through all possible starting positions for a substring of length 2 in s.
        # The loop runs from i = 0 to n-2.
        # For a string of length n, there are n-1 substrings of length 2.
        # Example: s = "abc", n=3. i iterates for 0, 1.
        #   i=0: s[0:2] = "ab"
        #   i=1: s[1:3] = "bc"
        # The range(n - 1) goes from 0 up to (n-1)-1 = n-2.
        for i in range(n - 1):
            # Extract the substring of length 2 from s starting at index i.
            # This is s[i:i+2], which corresponds to characters s[i] and s[i+1].
            # This operation is O(1) because the length of the slice is fixed at 2.
            substring_from_s = s[i : i + 2]
            
            # Check if this substring_from_s is present anywhere in reversed_s.
            # The 'in' operator for string containment in Python has an average
            # time complexity related to the lengths of the strings involved.
            # Specifically, O(len(reversed_s) * len(substring_from_s)) in worst case,
            # but often faster with algorithms like Knuth-Morris-Pratt or Boyer-Moore.
            # For fixed len(substring_from_s) = 2, this check is roughly O(n).
            if substring_from_s in reversed_s:
                # If found, we can immediately return True as per the problem statement.
                return True
                
        # If the loop completes without finding any such substring,
        # it means no substring of length 2 from s is present in reversed_s.
        return False