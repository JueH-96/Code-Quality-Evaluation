class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # Convert n and k to their binary representations
        n_bin = bin(n)[2:]
        k_bin = bin(k)[2:]

        # Pad the shorter binary string with leading zeros
        max_len = max(len(n_bin), len(k_bin))
        n_bin = n_bin.zfill(max_len)
        k_bin = k_bin.zfill(max_len)

        # Count the number of changes needed
        changes = 0
        for i in range(max_len):
            if n_bin[i] == '1' and k_bin[i] == '0':
                changes += 1
            elif n_bin[i] == '0' and k_bin[i] == '1':
                return -1

        return changes