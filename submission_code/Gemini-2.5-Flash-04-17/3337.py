class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        
        # Count the total number of occurrences of character c in string s.
        # Let this count be k.
        # Python's string count method is efficient for this.
        count_c = s.count(c)

        # If character 'c' appears 'k' times in the string 's',
        # any substring starting and ending with 'c' must correspond to
        # choosing two occurrences of 'c' from the 'k' occurrences, where
        # the starting occurrence appears at an index less than or equal to
        # the ending occurrence.
        
        # Let the indices of the occurrences of 'c' in 's' be p_1 < p_2 < ... < p_k.
        # A substring s[i:j+1] starts and ends with 'c' if s[i]=c, s[j]=c, and i <= j.
        # This means i must be p_a and j must be p_b for some 1 <= a <= k and 1 <= b <= k,
        # such that p_a <= p_b. Since the indices p_m are strictly increasing,
        # p_a <= p_b is equivalent to a <= b.
        # So, we need to count the number of pairs of indices (a, b) such that
        # 1 <= a <= k and a <= b <= k.
        # This count is equal to the sum 1 + 2 + ... + k, where the m-th term
        # represents the number of valid substrings ending at the m-th occurrence of 'c'
        # (these substrings can start at any of the first m occurrences of 'c').
        # The sum 1 + 2 + ... + k is given by the formula k * (k + 1) / 2.
        
        # Calculate the result using the formula count_c * (count_c + 1) / 2.
        # Use integer division // as the result must be an integer.
        # count_c * (count_c + 1) is always even, so integer division is appropriate.
        
        result = count_c * (count_c + 1) // 2
        
        return result