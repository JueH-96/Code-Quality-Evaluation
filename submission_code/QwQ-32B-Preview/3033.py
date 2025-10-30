class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        diff_indices = [i for i in range(n) if s1[i] != s2[i]]
        k = len(diff_indices)
        
        if k % 2 != 0:
            return -1  # Impossible to make s1 equal to s2
        
        # Group difference indices into sequences of consecutive differences
        sequences = []
        start = 0
        for i in range(1, k):
            if diff_indices[i] - diff_indices[i - 1] != 1:
                sequences.append(diff_indices[start:i])
                start = i
        sequences.append(diff_indices[start:])
        
        total_cost = 0
        for seq in sequences:
            l = len(seq)
            if l % 2 == 0:
                total_cost += l // 2
            else:
                # Use ceil(l / 2) operations of type 2 and one operation of type 1 for the remaining
                total_cost += (l // 2) + x
        
        # After fixing all sequences, check for remaining isolated differences
        # Since k is even and we've processed all differences, no need to handle separately
        
        return total_cost