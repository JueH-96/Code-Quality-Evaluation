import itertools

class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        
        def _get_overlap_merged_string(s1: str, s2: str) -> str:
            """
            Returns s1 with s2 appended, maximizing overlap where s1's suffix
            matches s2's prefix. This function assumes s1 comes before s2
            and only considers overlaps where s1 contributes its suffix and s2 its prefix.
            
            Example: _get_overlap_merged_string("abc", "bca") -> "abca" (overlap "bc")
            Example: _get_overlap_merged_string("aaabc", "bca") -> "aaabca" (overlap "bc")
            """
            max_overlap = 0
            # Iterate from largest possible overlap down to 1
            # k represents the length of the potential overlap
            for k in range(min(len(s1), len(s2)), 0, -1):
                # Check if the last k characters of s1 match the first k characters of s2
                if s1.endswith(s2[:k]):
                    max_overlap = k
                    break
            # Return s1 followed by the non-overlapping part of s2
            return s1 + s2[max_overlap:]

        def combine(s1: str, s2: str) -> str:
            """
            Returns the shortest string that contains both s1 and s2 as substrings.
            If there are multiple strings with the minimum length, returns the
            lexicographically smallest one.
            """
            # Case 1: One string already fully contains the other.
            # This is the shortest possible combined string.
            # Check s1 in s2 first. If true, s2 is the result.
            if s1 in s2:
                return s2
            # Check s2 in s1 next. If true, s1 is the result.
            if s2 in s1:
                return s1
            
            # Case 2: No full containment. We need to find the best overlap.
            # Try combining s1 and s2 in the order s1 then s2.
            # E.g., combine("abc", "bca") -> "abca" by overlapping "bc"
            res1 = _get_overlap_merged_string(s1, s2)
            
            # Try combining s1 and s2 in the order s2 then s1.
            # E.g., combine("bca", "abc") -> "bcabc" by overlapping "a"
            res2 = _get_overlap_merged_string(s2, s1)
            
            # Compare the two results based on length, then lexicographical order.
            if len(res1) < len(res2):
                return res1
            elif len(res2) < len(res1):
                return res2
            else: # Lengths are equal, return the lexicographically smaller one
                return min(res1, res2)

        # List of the three input strings
        strings = [a, b, c]
        
        # Initialize the answer with an empty string. The result from the first
        # permutation will set the initial answer.
        ans = "" 

        # Iterate through all 3! = 6 permutations of the input strings.
        # For each permutation (s1, s2, s3), we combine them sequentially:
        # first combine s1 and s2, then combine the result with s3.
        for p in itertools.permutations(strings):
            s1, s2, s3 = p[0], p[1], p[2]
            
            # Step 1: Combine the first two strings
            # This intermediate result optimally contains s1 and s2
            merged_s1_s2 = combine(s1, s2)
            
            # Step 2: Combine the intermediate result with the third string
            # This final result optimally contains s1, s2, and s3 in this specific order
            current_result = combine(merged_s1_s2, s3)
            
            # Update the overall best answer found so far
            if not ans: # If ans is empty, this is the first result
                ans = current_result
            elif len(current_result) < len(ans): # Current result is shorter
                ans = current_result
            elif len(current_result) == len(ans): # Current result has same length
                ans = min(ans, current_result) # Choose lexicographically smaller
                
        return ans