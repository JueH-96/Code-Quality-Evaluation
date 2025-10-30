from typing import List

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total_pairs = n // 2
        freq_D = {}
        
        # Compute frequency of differences
        for i in range(total_pairs):
            a = nums[i]
            b = nums[n - i - 1]
            D = abs(a - b)
            freq_D[D] = freq_D.get(D, 0) + 1
        
        # Initialize difference array for range updates
        range_counts = [0] * (k + 2)
        
        # Update the difference array for each pair
        for i in range(total_pairs):
            a = nums[i]
            b = nums[n - i - 1]
            max_end = max(k - b, b, k - a, a)
            range_counts[0] += 1
            if max_end + 1 <= k + 1:
                range_counts[max_end + 1] -= 1
            else:
                range_counts[k + 1] -= 1
        
        # Compute prefix sums to get the number of pairs that can achieve X with 1 change
        prefix = 0
        freq_X1 = [0] * (k + 1)
        for X in range(k + 1):
            prefix += range_counts[X]
            # freq_X1[X] is the number of pairs that can achieve X with 1 change
            # Subtract freq_D[X] to exclude pairs that can achieve X with 0 changes
            freq_X1[X] = prefix - freq_D.get(X, 0)
        
        # Calculate the score for each X and find the maximum score
        max_score = 0
        for X in range(k + 1):
            score = freq_D.get(X, 0) * 2 + freq_X1[X] * 1
            if score > max_score:
                max_score = score
        
        # Calculate the minimum total changes
        min_total_changes = total_pairs * 2 - max_score
        return min_total_changes