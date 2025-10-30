class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        min_ops = n  # worst case: delete all digits to get 0
        
        # Check if we can keep a single '0'
        if '0' in num:
            min_ops = n - 1
        
        # Check for endings: 00, 25, 50, 75
        endings = ['00', '25', '50', '75']
        
        for ending in endings:
            # Find the rightmost occurrence of the second digit
            second_digit = ending[1]
            second_pos = -1
            
            for i in range(n - 1, -1, -1):
                if num[i] == second_digit:
                    second_pos = i
                    break
            
            if second_pos == -1:
                continue
            
            # Find the rightmost occurrence of the first digit before second_pos
            first_digit = ending[0]
            first_pos = -1
            
            for i in range(second_pos - 1, -1, -1):
                if num[i] == first_digit:
                    first_pos = i
                    break
            
            if first_pos == -1:
                continue
            
            # Calculate operations needed
            # Delete all digits after second_pos and all digits between first_pos and second_pos
            ops = (n - 1 - second_pos) + (second_pos - first_pos - 1)
            min_ops = min(min_ops, ops)
        
        return min_ops