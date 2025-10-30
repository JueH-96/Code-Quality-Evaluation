class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        # Compress values to reduce state space
        unique_vals = sorted(set(nums))
        val_to_idx = {val: i for i, val in enumerate(unique_vals)}
        compressed_nums = [val_to_idx[num] for num in nums]
        
        n = len(nums)
        
        from functools import lru_cache
        
        @lru_cache(None)
        def solve(index, used_diff_pairs, last_val):
            if index == n:
                return 0
            
            # Option 1: Don't include nums[index]
            result = solve(index + 1, used_diff_pairs, last_val)
            
            # Option 2: Include nums[index]
            if last_val == -1:  # This is the first element
                result = max(result, 1 + solve(index + 1, used_diff_pairs, compressed_nums[index]))
            else:
                if compressed_nums[index] == last_val:
                    # Same as last, no additional diff pair
                    result = max(result, 1 + solve(index + 1, used_diff_pairs, compressed_nums[index]))
                else:
                    # Different from last, uses one more diff pair
                    if used_diff_pairs < k:
                        result = max(result, 1 + solve(index + 1, used_diff_pairs + 1, compressed_nums[index]))
            
            return result
        
        return solve(0, 0, -1)