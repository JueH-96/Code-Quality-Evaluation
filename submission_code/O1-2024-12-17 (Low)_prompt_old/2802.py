class Solution:
    def punishmentNumber(self, n: int) -> int:
        # Helper function to check if the square of a number can be partitioned
        # such that the sum of its partitions equals the original number.
        def can_partition(num_str: str, target: int) -> bool:
            # If we've matched the sum exactly and no digits remain, it's valid
            if target == 0 and not num_str:
                return True
            # If no digits remain but we haven't matched the sum, it's invalid
            if not num_str:
                return False
            
            # Try all possible splits for the next partition
            for i in range(1, len(num_str) + 1):
                part = int(num_str[:i])
                if part <= target:
                    if can_partition(num_str[i:], target - part):
                        return True
            return False
        
        total_sum = 0
        for i in range(1, n + 1):
            sq_str = str(i * i)
            if can_partition(sq_str, i):
                total_sum += i * i
        return total_sum