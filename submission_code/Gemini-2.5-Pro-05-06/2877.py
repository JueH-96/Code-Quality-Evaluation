from itertools import permutations

class Solution:
  def minimumString(self, a: str, b: str, c: str) -> str:
    
    # Defines merge(s1, s2)
    # Returns shortest string that effectively represents s1 followed by s2.
    # s1 and s2 may overlap, or one may contain the other.
    def merge(s1: str, s2: str) -> str:
        # Case 1: s1 already contains s2 as a substring.
        # If s1 contains s2, s1 itself satisfies the condition of having s1 and s2.
        # Example: s1="banana", s2="ana" -> "banana"
        if s2 in s1:
            return s1
        
        # Case 2: s2 already contains s1 as a substring.
        # If s2 contains s1, s2 itself satisfies the condition.
        # Example: s1="short", s2="veryshortstring" -> "veryshortstring"
        if s1 in s2:
            return s2
        
        # Case 3: Overlap s1's suffix with s2's prefix.
        # Find the longest suffix of s1 that matches a prefix of s2.
        # Example: s1="abcde", s2="cdefg" -> "abcdefg" (overlap "cde", k=3)
        max_overlap_len = 0
        # Iterate k (potential overlap length) from min(len(s1), len(s2)) down to 1.
        # (k=0 is the base case: no overlap, implies s1 + s2)
        for k_test in range(min(len(s1), len(s2)), 0, -1):
            # Check if suffix of s1 (length k_test) matches prefix of s2 (length k_test)
            if s1.endswith(s2[:k_test]): 
                max_overlap_len = k_test
                break
        # Append the non-overlapping part of s2 to s1
        return s1 + s2[max_overlap_len:]

    # Store the input strings in a list for easy permutation
    string_list = [a, b, c]
    
    # Initialize best_s. A safe initial value is the concatenation of all three strings,
    # as this is a valid superstring. Its length is an upper bound for the minimum length.
    best_s = a + b + c 

    # Iterate over all 3! = 6 permutations of the strings
    for p in permutations(string_list):
        s_p1, s_p2, s_p3 = p[0], p[1], p[2]
        
        # Form candidate string by merging in the permuted order:
        # First, merge s_p1 and s_p2. Let the result be merged_s1_s2.
        # Then, merge merged_s1_s2 with s_p3.
        merged_s1_s2 = merge(s_p1, s_p2)
        candidate_str = merge(merged_s1_s2, s_p3)
        
        # Update best_s if this candidate is better:
        # - If candidate_str is shorter than best_s.
        # - If candidate_str has the same length as best_s but is lexicographically smaller.
        if len(candidate_str) < len(best_s):
            best_s = candidate_str
        elif len(candidate_str) == len(best_s):
            if candidate_str < best_s: # Lexicographical comparison
                best_s = candidate_str
    
    return best_s