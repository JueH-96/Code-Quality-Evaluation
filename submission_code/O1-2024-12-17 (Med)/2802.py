class Solution:
    def punishmentNumber(self, n: int) -> int:
        
        def can_partition(num_str: str, idx: int, current_sum: int, target: int, memo) -> bool:
            # If we reach the exact sum and have used all digits, it's a valid partition.
            if current_sum == target and idx == len(num_str):
                return True
            
            # If we've exceeded the sum or used up digits without exact match, not valid.
            if current_sum > target or idx >= len(num_str):
                return False
            
            if (idx, current_sum) in memo:
                return memo[(idx, current_sum)]
            
            # Try all possible partitions from this index onward
            for end in range(idx + 1, len(num_str) + 1):
                val = int(num_str[idx:end])
                next_sum = current_sum + val
                # Only proceed if we haven't exceeded the target
                if next_sum <= target:
                    if can_partition(num_str, end, next_sum, target, memo):
                        memo[(idx, current_sum)] = True
                        return True
            
            memo[(idx, current_sum)] = False
            return False
        
        def is_good(i: int) -> bool:
            s = str(i * i)
            # Use memo to avoid repeated calculations in can_partition
            return can_partition(s, 0, 0, i, {})
        
        total = 0
        for i in range(1, n + 1):
            if is_good(i):
                total += i * i
        
        return total