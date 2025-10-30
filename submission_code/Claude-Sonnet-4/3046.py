class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        min_ops = n  # worst case: delete all digits to get "0"
        
        # Check if we can form "0" by keeping one '0'
        if '0' in num:
            min_ops = n - 1
        
        # Try to form endings: "00", "25", "50", "75"
        endings = ["00", "25", "50", "75"]
        
        for ending in endings:
            last_digit = ending[1]
            first_digit = ending[0]
            
            # Find rightmost occurrence of last digit
            last_pos = -1
            for i in range(n - 1, -1, -1):
                if num[i] == last_digit:
                    last_pos = i
                    break
            
            if last_pos == -1:
                continue
                
            # Find rightmost occurrence of first digit before last_pos
            first_pos = -1
            for i in range(last_pos - 1, -1, -1):
                if num[i] == first_digit:
                    first_pos = i
                    break
            
            if first_pos == -1:
                continue
            
            # Calculate operations needed
            # We keep digits at first_pos and last_pos, delete the rest
            ops = n - 2
            min_ops = min(min_ops, ops)
        
        return min_ops