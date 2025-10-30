class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # Convert n to binary and remove the '0b' prefix
        n_bin = bin(n)[2:]
        # Calculate the length of the binary representation of n
        n_len = len(n_bin)
        # Initialize the minimum number of changes to infinity
        min_changes = float('inf')
        # Iterate over all possible values of k
        for i in range(2**n_len):
            # Convert i to binary and pad it with zeros to the left to make its length equal to n_len
            k_bin = bin(i)[2:].zfill(n_len)
            # Initialize the number of changes to 0
            changes = 0
            # Iterate over the bits of k_bin
            for j in range(n_len):
                # If the j-th bit of k_bin is 1 and the j-th bit of n_bin is 0, increment the number of changes
                if k_bin[j] == '1' and n_bin[j] == '0':
                    changes += 1
            # If the number of changes is less than the current minimum, update the minimum
            min_changes = min(min_changes, changes)
        # If the minimum number of changes is infinity, return -1
        if min_changes == float('inf'):
            return -1
        # Otherwise, return the minimum number of changes
        return min_changes