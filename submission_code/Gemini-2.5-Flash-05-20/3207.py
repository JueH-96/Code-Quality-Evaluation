class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        
        # Step 1: Find the minimum length among the three strings.
        # This determines the maximum possible length of the common prefix we can check.
        min_len = min(len(s1), len(s2), len(s3))
        
        # Step 2: Initialize lcp_len (length of the longest common prefix).
        # This will store the length of the common prefix found.
        lcp_len = 0
        
        # Step 3: Iterate through the strings character by character up to min_len
        # to find the length of the longest common prefix.
        for i in range(min_len):
            # Check if characters at the current index 'i' are equal in all three strings.
            if s1[i] == s2[i] and s1[i] == s3[i]:
                lcp_len += 1  # If they match, extend the common prefix
            else:
                # If a mismatch is found, the common prefix ends at the previous character.
                # Break the loop as we've found the true LCP length.
                break
                
        # Step 4: If lcp_len is 0, it means the strings do not share even a single common
        # starting character. In such a scenario, it's impossible to make them equal
        # to a non-empty string. As per example 2, this case should return -1.
        if lcp_len == 0:
            return -1
            
        # Step 5: Calculate the total number of operations.
        # The goal is to make all strings equal to their LCP (s1[0...lcp_len-1]).
        # For each string, the operations needed are its current length minus the LCP length.
        # This represents deleting the excess characters from the right.
        operations = (len(s1) - lcp_len) + \
                     (len(s2) - lcp_len) + \
                     (len(s3) - lcp_len)
                     
        # Step 6: Return the calculated minimum total operations.
        return operations