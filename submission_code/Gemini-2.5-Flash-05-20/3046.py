class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        
        # Initialize min_ops.
        # The worst case is deleting all digits to get "0", which takes n operations.
        # If '0' exists in num, we can form "0" by deleting n-1 digits (keeping one '0').
        # This is generally better than deleting all digits (n operations).
        min_ops = n 
        if '0' in num:
            min_ops = n - 1
            
        # Target suffixes for divisibility by 25: "00", "25", "50", "75"
        target_suffixes = [('0', '0'), ('2', '5'), ('5', '0'), ('7', '5')]
        
        for d1, d2 in target_suffixes:
            # Step 1: Find the rightmost occurrence of d2 (the second digit of the target suffix)
            idx2 = -1
            for i in range(n - 1, -1, -1): # Iterate from right to left
                if num[i] == d2:
                    idx2 = i
                    break
            
            if idx2 == -1:
                # If d2 is not found in the string, this target suffix cannot be formed.
                continue
            
            # Step 2: Find the rightmost occurrence of d1 (the first digit) before idx2
            idx1 = -1
            for j in range(idx2 - 1, -1, -1): # Iterate from idx2-1 down to 0
                if num[j] == d1:
                    idx1 = j
                    break
            
            if idx1 == -1:
                # If d1 is not found before idx2, this target suffix cannot be formed
                # using the current idx2.
                continue
            
            # Both d1 and d2 are found at valid positions (idx1 < idx2).
            # Calculate the number of operations required to form this specific suffix:
            # Operations = (digits after idx2) + (digits between idx1 and idx2)
            # 1. Digits after idx2: n - 1 - idx2
            # 2. Digits between idx1 and idx2: idx2 - 1 - idx1
            # Total operations = (n - 1 - idx2) + (idx2 - 1 - idx1)
            # This simplifies to n - 2 - idx1.
            
            current_ops = (n - 1 - idx2) + (idx2 - 1 - idx1)
            
            # Update the minimum operations found so far
            min_ops = min(min_ops, current_ops)
            
        return min_ops