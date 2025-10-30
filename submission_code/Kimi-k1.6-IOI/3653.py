class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        min_residue = [float('inf')] * k
        min_residue[0] = 0  # prefix_sum[0] is 0, residue 0 mod k is 0
        max_sum = float('-inf')
        
        for j in range(1, n + 1):
            r = j % k
            if min_residue[r] != float('inf'):
                current_sum = prefix_sum[j] - min_residue[r]
                if current_sum > max_sum:
                    max_sum = current_sum
            # Update the minimum for the current residue
            current_r = j % k
            if prefix_sum[j] < min_residue[current_r]:
                min_residue[current_r] = prefix_sum[j]
        
        return max_sum