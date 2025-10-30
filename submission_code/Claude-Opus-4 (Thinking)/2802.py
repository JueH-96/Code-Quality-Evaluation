class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_partition(square_str, target, index=0, current_sum=0):
            # Base case: if we've processed all digits
            if index == len(square_str):
                return current_sum == target
            
            # Try all possible partitions starting from index
            for i in range(index, len(square_str)):
                # Get the substring from index to i (inclusive)
                substring = square_str[index:i+1]
                # Convert to integer
                num = int(substring)
                # Recursively check if we can reach target with this partition
                if can_partition(square_str, target, i+1, current_sum + num):
                    return True
            
            return False
        
        result = 0
        for i in range(1, n + 1):
            square = i * i
            square_str = str(square)
            if can_partition(square_str, i):
                result += square
        
        return result