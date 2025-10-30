class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        # Special numbers to check for divisibility by 25
        special_ends = ["00", "25", "50", "75"]
        
        # Initialize the minimum operations to a large number
        min_operations = float('inf')
        
        # Check for each pair of special endings
        for ends in special_ends:
            last_digit = ends[1]
            second_last_digit = ends[0]
            
            # Find the last occurrence of the last_digit
            last_pos = -1
            for i in range(n-1, -1, -1):
                if num[i] == last_digit:
                    last_pos = i
                    break
            
            if last_pos == -1:
                continue  # No such last_digit found, skip this pair
            
            # Find the second last digit before the last_pos
            second_last_pos = -1
            for i in range(last_pos-1, -1, -1):
                if num[i] == second_last_digit:
                    second_last_pos = i
                    break
            
            if second_last_pos == -1:
                continue  # No such second_last_digit found before last_pos, skip this pair
            
            # Calculate operations needed to make the number end with this pair
            # We need to remove all digits after last_pos and between second_last_pos and last_pos
            operations = (n - last_pos - 1) + (last_pos - second_last_pos - 1)
            min_operations = min(min_operations, operations)
        
        # If no special number can be formed, the answer is to delete all digits except one
        if min_operations == float('inf'):
            return n - 1
        
        return min_operations