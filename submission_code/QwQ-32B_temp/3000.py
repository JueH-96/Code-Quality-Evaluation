class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        sorted_list = sorted((val, idx) for idx, val in enumerate(nums))
        min_diff = float('inf')
        n = len(sorted_list)
        window_size = 200  # Adjust this window size based on expected input characteristics
        
        for i in range(n):
            current_val, current_idx = sorted_list[i]
            # Check next elements up to window_size ahead
            end = min(i + window_size + 1, n)
            for j in range(i + 1, end):
                other_val, other_idx = sorted_list[j]
                val_diff = other_val - current_val
                if val_diff >= min_diff:
                    break  # Further elements will have larger differences
                if abs(current_idx - other_idx) >= x:
                    if val_diff < min_diff:
                        min_diff = val_diff
        
        return min_diff if min_diff != float('inf') else 0