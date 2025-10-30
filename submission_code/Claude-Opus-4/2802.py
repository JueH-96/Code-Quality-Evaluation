class Solution:
    def punishmentNumber(self, n: int) -> int:
        def canPartition(num_str, target, index=0, current_sum=0):
            # Base case: if we've processed all digits
            if index == len(num_str):
                return current_sum == target
            
            # Try all possible partitions starting from current index
            for i in range(index, len(num_str)):
                # Get the substring from index to i (inclusive)
                substring = num_str[index:i+1]
                # Convert to integer
                value = int(substring)
                
                # If adding this value doesn't exceed target, try it
                if current_sum + value <= target:
                    if canPartition(num_str, target, i+1, current_sum + value):
                        return True
            
            return False
        
        punishment_sum = 0
        
        # Check each number from 1 to n
        for i in range(1, n + 1):
            square = i * i
            square_str = str(square)
            
            # Check if the square can be partitioned to sum to i
            if canPartition(square_str, i):
                punishment_sum += square
        
        return punishment_sum