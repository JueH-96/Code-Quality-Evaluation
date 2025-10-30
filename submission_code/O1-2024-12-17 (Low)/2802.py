class Solution:
    def punishmentNumber(self, n: int) -> int:

        # Helper function to check if we can partition the string representation
        # of num_str into contiguous substrings that sum up to target.
        def can_partition(num_str: str, target: int) -> bool:
            # If we've used up all digits and exactly matched the target, return True
            if not num_str:
                return target == 0

            # Try all possible splits
            for i in range(1, len(num_str) + 1):
                part = int(num_str[:i])
                # If this partition is not greater than the target, explore further
                if part <= target:
                    if can_partition(num_str[i:], target - part):
                        return True
            
            return False

        total_sum = 0

        # Check each i from 1 to n
        for i in range(1, n + 1):
            sq_str = str(i * i)
            # If i^2 can be partitioned into substrings summing to i, add i^2 to total_sum
            if can_partition(sq_str, i):
                total_sum += i * i

        return total_sum