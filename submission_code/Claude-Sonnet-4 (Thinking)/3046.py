class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        min_ops = n  # Worst case: delete all digits to get 0
        
        # Check if we can make the number "0" by keeping one '0'
        if '0' in num:
            min_ops = n - 1
        
        # Define the target suffixes for numbers divisible by 25
        targets = [('0', '0'), ('2', '5'), ('5', '0'), ('7', '5')]
        
        for first_digit, last_digit in targets:
            # Find the rightmost occurrence of the last digit
            last_pos = -1
            for i in range(n-1, -1, -1):
                if num[i] == last_digit:
                    last_pos = i
                    break
            
            if last_pos == -1:
                continue
            
            # Find the rightmost occurrence of the first digit before last_pos
            first_pos = -1
            for j in range(last_pos-1, -1, -1):
                if num[j] == first_digit:
                    first_pos = j
                    break
            
            if first_pos == -1:
                continue
            
            # Calculate deletions: digits between first_pos and last_pos + digits after last_pos
            ops = (last_pos - first_pos - 1) + (n - 1 - last_pos)
            min_ops = min(min_ops, ops)
        
        return min_ops