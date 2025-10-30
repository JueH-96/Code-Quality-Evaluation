import numpy as np

class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n = len(s)
        m = len(pattern)

        # Constraints state m < n, so m <= n-1. n-m >= 1.
        # A substring of length m exists if n >= m.
        # The loop range for i (0 to n-m) is valid.

        # We need the number of mismatches between s[i:i+m] and pattern
        # to be at most 1.
        # This is equivalent to the number of matches being at least m - 1.
        # Number of matches = sum_{j=0}^{m-1} [s[i+j] == pattern[j]]

        # We can compute the number of matches for all possible starting indices i
        # using FFT for character-by-character matching and summing up.
        # Let S_c[k] = 1 if s[k] == c, else 0.
        # Let P_c[j] = 1 if pattern[j] == c, else 0.
        # Number of matches for char c at index i = sum_{j=0}^{m-1} S_c[i+j] * P_c[j].
        # This sum is the cross-correlation of S_c and P_c evaluated at index i.
        # Cross-correlation (A * B)[i] = sum_j A[i+j] B[j].
        # Using FFT, cross-correlation (A * B) is computed as IFFT(FFT(A) * conj(FFT(B))).
        # Or equivalently, (A * B)[i] = (A * B_rev)[i + len(B) - 1],
        # where B_rev[k] = B[len(B)-1-k].
        # Here A=S_c (len n), B=P_c (len m). B_rev=P_c[::-1].
        # sum_{j=0}^{m-1} S_c[i+j] * P_c[j] = (S_c * P_c_rev)[i + m - 1].

        # Determine the length L for FFT. Must be a power of 2
        # and >= n + m - 1.
        # The required length for convolution result is n + m - 1.
        # FFT length must be >= this value and a power of 2.
        required_len = n + m - 1
        fft_len = 1
        while fft_len < required_len:
            fft_len *= 2

        # Array to store the sum of matches for each convolution index l.
        # total_match_counts_conv_indices[l] = sum_c (S_c * P_c_rev)[l]
        # We only need values for l from m-1 to n-1.
        # This array will store the sum of match indicators for each character
        # at the corresponding convolution index.
        total_match_counts_conv_indices = np.zeros(fft_len, dtype=float)

        # Iterate through the alphabet 'a' to 'z'.
        # This ensures we cover all possible characters that could contribute to a match.
        for char_code in range(ord('a'), ord('z') + 1):
            c = chr(char_code)

            # Create S_c array (1.0 if s[k] == c, 0.0 otherwise)
            S_c = np.zeros(n, dtype=float)
            for k in range(n):
                if s[k] == c:
                    S_c[k] = 1.0

            # Create P_c array (1.0 if pattern[j] == c, 0.0 otherwise)
            # Create reversed P_c array (P_c_rev)
            P_c = np.zeros(m, dtype=float)
            for j in range(m):
                if pattern[j] == c:
                    P_c[j] = 1.0

            P_c_rev = P_c[::-1] # Reverse the numpy array

            # Pad arrays to fft_len
            S_c_padded = np.pad(S_c, (0, fft_len - n), 'constant')
            P_c_rev_padded = np.pad(P_c_rev, (0, fft_len - m), 'constant')

            # Compute FFTs
            fft_S_c = np.fft.fft(S_c_padded)
            fft_P_c_rev = np.fft.fft(P_c_rev_padded)

            # Multiply in frequency domain to get FFT of convolution (S_c * P_c_rev)
            fft_conv_c = fft_S_c * fft_P_c_rev

            # Compute inverse FFT to get the convolution result in time domain
            conv_c = np.fft.ifft(fft_conv_c)

            # Add the real part of the convolution result.
            # The theoretical convolution result is real, but IFFT might produce
            # small imaginary components due to floating point precision.
            total_match_counts_conv_indices += conv_c.real

        # Now, iterate through the relevant convolution indices l to find the smallest
        # corresponding starting index i that satisfies the condition.
        # The convolution index l relates to the starting index i by l = i + m - 1.
        # We are interested in i from 0 to n-m.
        # This corresponds to l from 0 + m - 1 = m - 1 to (n-m) + m - 1 = n - 1.
        # So we iterate l from m-1 up to n-1 (inclusive).
        # range(m - 1, n) goes from m-1 up to n-1.

        # Use a small tolerance for floating point comparisons.
        # The number of matches should be an integer >= m - 1.
        # We check if the computed float value is close to an integer >= m - 1.
        # Specifically, we check if num_matches_float >= m - 1 - epsilon.
        # This checks if the number of mismatches (m - num_matches_float) <= 1 + epsilon.
        # An epsilon less than 0.5 is typically sufficient to distinguish integers.
        epsilon = 0.1

        # Iterate through the range of convolution indices that correspond
        # to valid starting indices in s.
        for l in range(m - 1, n):
            # The starting index i in s corresponding to convolution index l is:
            i = l - (m - 1)

            # The total number of matches for substring s[i:i+m] with pattern is
            # given by total_match_counts_conv_indices[l].
            num_matches_float = total_match_counts_conv_indices[l]

            # We need the number of matches to be >= m - 1 (which means at most 1 mismatch).
            # Check if the computed number of matches is approximately >= m - 1.
            if num_matches_float >= m - 1.0 - epsilon:
                 # Found the smallest starting index i that satisfies the condition.
                 return i

        # If the loop finishes without returning, no such starting index was found.
        return -1