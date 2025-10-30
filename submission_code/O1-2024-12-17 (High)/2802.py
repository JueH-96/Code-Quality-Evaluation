class Solution:
    def punishmentNumber(self, n: int) -> int:
        # Helper function to check if we can partition the square's string representation
        # into contiguous substrings that sum to the original number i.
        def can_partition(num_str: str, target: int, start: int, memo) -> bool:
            if (start, target) in memo:
                return memo[(start, target)]
            if start == len(num_str):
                return target == 0
            
            # Try all possible partitions starting at 'start'
            current_sum = 0
            for end in range(start + 1, len(num_str) + 1):
                val = int(num_str[start:end])
                if val <= target:
                    if can_partition(num_str, target - val, end, memo):
                        memo[(start, target)] = True
                        return True
            memo[(start, target)] = False
            return False
        
        total = 0
        # Check each integer i from 1 to n
        for i in range(1, n + 1):
            square_str = str(i * i)
            # If the square can be partitioned into substrings summing to i, add i*i to total
            if can_partition(square_str, i, 0, {}):
                total += i * i
        
        return total