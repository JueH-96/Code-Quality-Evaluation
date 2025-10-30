class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        min_deletions = n  # worst case: delete all digits to get 0
        
        # Check if we can make "0" by keeping one '0'
        if '0' in num:
            min_deletions = n - 1
        
        # Check for each pattern that makes a number divisible by 25
        patterns = [('0', '0'), ('2', '5'), ('5', '0'), ('7', '5')]
        
        for first_digit, second_digit in patterns:
            # Find the rightmost occurrence of the second digit (last digit of the pattern)
            second_pos = -1
            for i in range(n - 1, -1, -1):
                if num[i] == second_digit:
                    second_pos = i
                    break
            
            if second_pos == -1:
                continue
            
            # Find the rightmost occurrence of the first digit before second_pos
            first_pos = -1
            for i in range(second_pos - 1, -1, -1):
                if num[i] == first_digit:
                    first_pos = i
                    break
            
            if first_pos == -1:
                continue
            
            # Calculate the number of deletions needed
            # Delete all digits between first_pos and second_pos, and all digits after second_pos
            deletions = (second_pos - first_pos - 1) + (n - second_pos - 1)
            min_deletions = min(min_deletions, deletions)
        
        return min_deletions