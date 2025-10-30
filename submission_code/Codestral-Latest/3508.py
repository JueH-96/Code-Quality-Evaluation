class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # Convert n and k to their binary representations
        bin_n = bin(n)[2:]
        bin_k = bin(k)[2:]

        # Pad the shorter binary string with leading zeros
        max_len = max(len(bin_n), len(bin_k))
        bin_n = bin_n.zfill(max_len)
        bin_k = bin_k.zfill(max_len)

        # Count the number of changes needed
        changes = 0
        for bit_n, bit_k in zip(bin_n, bin_k):
            if bit_n == '1' and bit_k == '0':
                changes += 1
            elif bit_n == '0' and bit_k == '1':
                return -1  # It's impossible to make n equal to k

        return changes