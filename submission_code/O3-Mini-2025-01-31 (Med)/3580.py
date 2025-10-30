import numpy as np

class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        # lengths
        n, m = len(s), len(pattern)
        # If m > n (should not happen as per constraints), return -1.
        if m > n:
            return -1

        # We'll perform convolution for each letter.
        # The convolution length: n + m - 1.
        conv_length = n + m - 1
        # For FFT efficiency, choose the next power of two.
        L = 1 << (conv_length - 1).bit_length()

        # Initialize an array to accumulate matching counts.
        total_conv = np.zeros(L, dtype=np.float64)
        
        # For each letter in the alphabet, create indicator arrays and accumulate convolution.
        for ch in "abcdefghijklmnopqrstuvwxyz":
            # Create indicator arrays for s and pattern.
            # We use np.frombuffer with np.uint8 if needed, but simpler is to build a list.
            A = np.array([1.0 if c == ch else 0.0 for c in s], dtype=np.float64)
            B = np.array([1.0 if c == ch else 0.0 for c in pattern], dtype=np.float64)
            # Reverse B for convolution matching technique.
            B = B[::-1]
            
            # Compute FFTs on zero-padded arrays.
            fft_A = np.fft.rfft(A, n=L)
            fft_B = np.fft.rfft(B, n=L)
            # Multiply in frequency domain and inverse FFT to get convolution.
            conv = np.fft.irfft(fft_A * fft_B, n=L)
            # Accumulate. We only need the first conv_length values.
            total_conv[:conv_length] += conv[:conv_length]

        # Now, for each valid window (starting index i from 0 to n-m),
        # the dot product (number of matches) is located at index i+m-1.
        # We want mismatches = m - matches to be <= 1.
        # Loop through possible starting indices.
        for i in range(n - m + 1):
            # Due to floating point arithmetic, we round the match count.
            matches = int(round(total_conv[i + m - 1]))
            if m - matches <= 1:
                return i
        return -1

# Example test cases:
if __name__ == '__main__':
    sol = Solution()
    print(sol.minStartingIndex("abcdefg", "bcdffg"))    # Output: 1
    print(sol.minStartingIndex("ababbababa", "bacaba"))   # Output: 4
    print(sol.minStartingIndex("abcd", "dba"))            # Output: -1
    print(sol.minStartingIndex("dde", "d"))               # Output: 0