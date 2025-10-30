class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        """
        Calculates the minimum number of operations to make the array nums equal to target.
        An operation consists of selecting a subarray and incrementing or decrementing it by 1.
        """
        
        # Let diff[i] = target[i] - nums[i]. The problem is equivalent to making the
        # diff array all zeros by adding/subtracting 1 from subarrays.
        
        # Consider a new array `changes` where `changes[i] = diff[i] - diff[i-1]`
        # (with diff[-1] = 0). Making `diff` all zeros is equivalent to making
        # `changes` all zeros.
        
        # An operation of +/- 1 on diff[i..j] (where j < n-1) is equivalent to
        # changing changes[i] by +/- 1 and changes[j+1] by -/+ 1.
        # An operation on diff[i..n-1] changes only changes[i].
        
        # Let `p_sum` be the sum of all positive values in `changes`, and `n_sum` be
        # the sum of absolute values of all negative values in `changes`.
        # `p_sum` is the total amount to be decreased, `n_sum` is the total to be increased.
        
        # We can pair up decreases and increases. The number of such "paired"
        # operations is min(p_sum, n_sum). The remaining `abs(p_sum - n_sum)` must be
        # handled by "unpaired" operations on suffixes.
        # Total operations = min(p_sum, n_sum) + abs(p_sum - n_sum) = max(p_sum, n_sum).

        p_sum = 0  # Total positive changes needed
        n_sum = 0  # Total negative changes needed (as a positive value)
        
        prev_diff = 0
        for i in range(len(nums)):
            current_diff = target[i] - nums[i]
            change = current_diff - prev_diff
            
            if change > 0:
                p_sum += change
            else:
                n_sum -= change # This is equivalent to n_sum += abs(change)
                
            prev_diff = current_diff
            
        return max(p_sum, n_sum)