import numpy as np

class Solution:
    """
    This class provides a method to find the smallest starting index of a substring
    in 's' that is almost equal to 'pattern'. A substring x is almost equal to y 
    if you can change at most one character in x to make it identical to y.
    This is equivalent to saying the Hamming distance between x and y is at most 1.
    """
    def minStartingIndex(self, s: str, pattern: str) -> int:
        """
        Finds the smallest starting index using FFT-based convolution to efficiently
        calculate the number of matching characters between all relevant substrings 
        of 's' and the 'pattern'.

        Args:
          s: The main string.
          pattern: The pattern string.

        Returns:
          The smallest starting index i such that the substring s[i:i+len(pattern)] 
          is almost equal to pattern (Hamming distance <= 1), or -1 if no such 
          index exists.
        """
        n = len(s)
        m = len(pattern)

        # Edge case: The pattern cannot be longer than the string s.
        if m > n:
            return -1
        
        # Trivial case: If the pattern length is 1. 
        # Any single character substring s[i] can be made equal to pattern[0] 
        # with at most one change (0 changes if s[i] == pattern[0], 1 change if different).
        # Thus, any s[i] is almost equal to pattern. The smallest starting index is 0.
        # We assume n >= 1 based on constraints, so s is not empty if m=1.
        if m == 1:
             return 0

        # --- FFT Based Approach for Hamming Distance Calculation ---

        # Calculate the required FFT length. It must be a power of 2 and 
        # greater than or equal to n + m - 1 to avoid circular convolution issues.
        N = 1
        while N < n + m - 1:
            N *= 2

        # Initialize an array to store the total number of matching characters 
        # for each possible starting index `i` (from 0 to n-m). The size is n - m + 1.
        total_matches = np.zeros(n - m + 1, dtype=int) 

        # Convert strings to numeric arrays (e.g., 'a'->0, 'b'->1, ... 'z'->25)
        # This simplifies creating binary indicator arrays needed for the FFT approach.
        # Specify dtype=int for clarity.
        s_numeric = np.array([ord(c) - ord('a') for c in s], dtype=int)
        p_numeric = np.array([ord(c) - ord('a') for c in pattern], dtype=int)
        
        # Reverse the pattern. The convolution theorem relates element-wise multiplication 
        # in the frequency domain to convolution in the time domain. For pattern matching 
        # via correlation, one of the sequences needs to be reversed.
        p_rev_numeric = p_numeric[::-1]

        # Iterate through each possible character in the alphabet ('a' through 'z')
        # represented by their character codes 0 through 25.
        for char_code in range(26):
            # Create binary indicator arrays for the current character `c`.
            # s_c[k] = 1 if s[k] == c, else 0
            s_c = (s_numeric == char_code).astype(int)
            # p_c_rev[k] = 1 if the character at index `m-1-k` in the original pattern is `c`, else 0
            p_c_rev = (p_rev_numeric == char_code).astype(int)

            # Optimization: If the current character `c` does not exist in the pattern, 
            # it cannot contribute to any matches. Skip the expensive FFT calculation for this character.
            # `np.any()` is generally faster than `np.sum() > 0` for boolean arrays.
            if not np.any(p_c_rev): 
                 continue

            # Pad the binary indicator arrays with zeros to the calculated FFT length N.
            s_c_padded = np.pad(s_c, (0, N - n))
            p_c_rev_padded = np.pad(p_c_rev, (0, N - m))

            # --- Perform Convolution using FFT ---
            # 1. Transform the padded signals to the frequency domain using FFT.
            fft_s_c = np.fft.fft(s_c_padded)
            fft_p_c_rev = np.fft.fft(p_c_rev_padded)
            
            # 2. Multiply the results element-wise in the frequency domain. 
            #    This corresponds to convolution in the time domain.
            convolution_fft = fft_s_c * fft_p_c_rev
            
            # 3. Transform the product back to the time domain using inverse FFT.
            convolution = np.fft.ifft(convolution_fft)

            # --- Extract Match Counts ---
            # The real part of the convolution result at index `k = i + m - 1` gives the 
            # number of positions `j` where s[i+j] == pattern[j] == c (the current character).
            # We need these counts for starting indices `i` from 0 to n - m.
            # These correspond to indices `m-1` to `n-1` in the full convolution result array.
            
            # Extract the relevant segment of the convolution result.
            # Use `.real` to discard negligible imaginary parts from floating point inaccuracies.
            # Round to the nearest integer to correct potential small floating point errors. 
            # The theoretical result of this convolution should be integer counts.
            match_counts_c = np.round(convolution[m-1 : n].real).astype(int)
            
            # Add the match counts for this character `c` to the overall total matches array.
            # We verify the length just as a safeguard, although it should be correct if N and slicing are right.
            if len(match_counts_c) == n - m + 1:
                 total_matches += match_counts_c
            else:
                 # This block indicates a potential logic error in indexing or size calculation.
                 # In a production scenario, logging or raising an error might be appropriate here.
                 # For this problem, we assume the calculations yield the correct length.
                 pass 


        # --- Find the Smallest Index Satisfying the Condition ---
        # Iterate through all possible starting indices `i` from 0 to n-m.
        for i in range(n - m + 1):
            # Calculate the Hamming distance between s[i:i+m] and pattern.
            # Hamming Distance = (Total length) - (Number of matching characters).
            hamming_dist = m - total_matches[i]
            
            # Check if the Hamming distance is at most 1, which is the definition of "almost equal".
            if hamming_dist <= 1:
                # Found the smallest starting index `i` that satisfies the condition. Return it.
                return i

        # If the loop completes without finding any index where the substring is almost equal to the pattern.
        return -1