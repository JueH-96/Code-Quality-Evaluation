class Solution:
    def punishmentNumber(self, n: int) -> int:
        total_punishment_sum = 0

        # This helper function checks if the string representation of a number (s_num_str)
        # can be partitioned into contiguous substrings whose integer values sum up to target_sum.
        # It uses a backtracking/recursive approach.
        def can_partition_sum(s_num_str: str, target_sum: int) -> bool:
            # Base case 1: If target_sum is 0 and there are no more characters in the string,
            # it means we have successfully partitioned the string and the sum matches.
            if target_sum == 0 and not s_num_str:
                return True
            
            # Base case 2: If target_sum is negative (sum exceeded target) or if there are
            # no more characters in the string but target_sum is not 0 (ran out of digits
            # before reaching target), then this path is invalid.
            if target_sum < 0 or not s_num_str:
                return False
            
            # Recursive step: Iterate through all possible lengths for the first substring.
            # 'i' represents the ending index (exclusive) for the current prefix substring.
            for i in range(1, len(s_num_str) + 1):
                current_val_str = s_num_str[:i]
                current_val = int(current_val_str) # Convert substring to integer
                
                # Optimization: If the current value itself is already greater than the
                # remaining target sum, then adding any future values will only make
                # the sum even larger. So, this path cannot lead to a solution.
                # Since 'i' is increasing, subsequent 'current_val_str' will be
                # longer and thus 'current_val' will be larger or equal, so we can break.
                if current_val > target_sum:
                    break
                
                # Recursively call the function for the rest of the string (s_num_str[i:])
                # and reduce the target sum by the current value.
                # If this recursive call returns True, it means a valid partition was found.
                if can_partition_sum(s_num_str[i:], target_sum - current_val):
                    return True
            
            # If no combination of partitions starting from the current point leads to the target sum,
            # return False.
            return False

        # Iterate through all integers 'i' from 1 to n (inclusive).
        for i in range(1, n + 1):
            square = i * i
            s_square = str(square) # Convert the square to its string representation
            
            # Check if the square's string representation can be partitioned
            # such that the sum of parts equals 'i'.
            if can_partition_sum(s_square, i):
                total_punishment_sum += square # If it can, add square to the total sum
                
        return total_punishment_sum