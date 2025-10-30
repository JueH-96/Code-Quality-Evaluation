class Solution:
    def minimumOperations(self, num: str) -> int:
        # Check if the original number is divisible by 25
        if int(num) % 25 == 0:
            return 0
        
        target_pairs = ['00', '25', '50', '75']
        min_deletions = float('inf')
        n = len(num)
        
        for pair in target_pairs:
            p, q = pair[0], pair[1]
            # Find the rightmost position of the second digit 'q'
            pos_q = -1
            for i in range(n-1, -1, -1):
                if num[i] == q:
                    pos_q = i
                    break
            if pos_q == -1:
                continue  # Pair not possible
            # Find the leftmost position of the first digit 'p' before pos_q
            pos_p = -1
            for i in range(pos_q-1, -1, -1):
                if num[i] == p:
                    pos_p = i
                    break
            if pos_p == -1:
                continue  # Pair not possible
            # Calculate deletions
            deletions = n - pos_p - 2
            if deletions < min_deletions:
                min_deletions = deletions
        
        # Check if there's at least one '0' in num
        if '0' in num:
            deletions_zero = n - 1
            if deletions_zero < min_deletions:
                min_deletions = deletions_zero
        else:
            # If no '0', need to delete all digits to make '0'
            deletions_all = n
            if deletions_all < min_deletions:
                min_deletions = deletions_all
        
        return min_deletions