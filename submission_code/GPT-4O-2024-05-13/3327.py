from typing import List

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        ones_positions = [i for i, num in enumerate(nums) if num == 1]
        num_ones = len(ones_positions)
        
        if num_ones >= k:
            # If there are already enough ones, we just need to count the minimum moves to pick them
            min_moves = float('inf')
            for i in range(num_ones - k + 1):
                mid = i + k // 2
                median = ones_positions[mid]
                moves = sum(abs(ones_positions[j] - (median - (k // 2 - (j - i)))) for j in range(i, i + k))
                min_moves = min(min_moves, moves)
            return min_moves
        
        # If there are not enough ones, we need to use maxChanges to create ones
        remaining_ones = k - num_ones
        min_moves = float('inf')
        
        for i in range(n):
            if nums[i] == 1:
                moves = 0
                changes = 0
                picked_ones = 1
                left = i - 1
                right = i + 1
                
                while picked_ones < k:
                    if changes < maxChanges:
                        if left >= 0 and nums[left] == 0:
                            nums[left] = 1
                            changes += 1
                            picked_ones += 1
                            moves += 1
                            left -= 1
                        elif right < n and nums[right] == 0:
                            nums[right] = 1
                            changes += 1
                            picked_ones += 1
                            moves += 1
                            right += 1
                        else:
                            break
                    else:
                        if left >= 0 and nums[left] == 1:
                            moves += 1
                            picked_ones += 1
                            left -= 1
                        elif right < n and nums[right] == 1:
                            moves += 1
                            picked_ones += 1
                            right += 1
                        else:
                            break
                
                if picked_ones == k:
                    min_moves = min(min_moves, moves)
        
        return min_moves