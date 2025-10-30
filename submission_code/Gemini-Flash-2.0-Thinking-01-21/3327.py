import math

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        min_total_moves = float('inf')
        for alice_index in range(n):
            current_nums = list(nums)
            initial_ones_at_alice_index = 0
            if current_nums[alice_index] == 1:
                initial_ones_at_alice_index = 1
                current_nums[alice_index] = 0
            ones_to_collect = k - initial_ones_at_alice_index
            if ones_to_collect <= 0:
                min_total_moves = 0
                continue
                
            costs_initial_ones = []
            costs_zeros = []
            initial_ones_indices = []
            zero_indices = []
            
            for i in range(n):
                if i != alice_index:
                    if current_nums[i] == 1:
                        costs_initial_ones.append(abs(i - alice_index))
                        initial_ones_indices.append(i)
                    else:
                        costs_zeros.append(1 + abs(i - alice_index))
                        zero_indices.append(i)
                        
            costs_initial_ones.sort()
            costs_zeros.sort()
            
            min_moves_for_alice_index = float('inf')
            
            for num_changes in range(min(ones_to_collect, maxChanges) + 1):
                ones_from_initial = ones_to_collect - num_changes
                if ones_from_initial >= 0 and ones_from_initial <= len(costs_initial_ones):
                    current_moves = 0
                    if num_changes > 0:
                        current_moves += sum(costs_zeros[:num_changes])
                    if ones_from_initial > 0:
                        current_moves += sum(costs_initial_ones[:ones_from_initial])
                    min_moves_for_alice_index = min(min_moves_for_alice_index, current_moves)
                    
            if min_moves_for_alice_index != float('inf'):
                min_total_moves = min(min_total_moves, min_moves_for_alice_index)
                
        return min_total_moves if min_total_moves != float('inf') else 0

from typing import List

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        min_total_moves = float('inf')
        for aliceIndex in range(n):
            initial_ones_at_aliceIndex = 1 if nums[aliceIndex] == 1 else 0
            ones_needed = k - initial_ones_at_aliceIndex
            if ones_needed <= 0:
                min_total_moves = 0
                continue
                
            costs_initial_ones = []
            costs_zeros = []
            initial_ones_count = 0
            zero_count = 0
            
            for i in range(n):
                if i != aliceIndex:
                    if nums[i] == 1:
                        costs_initial_ones.append(abs(i - aliceIndex))
                        initial_ones_count += 1
                    else:
                        costs_zeros.append(1 + abs(i - aliceIndex))
                        zero_count += 1
                        
            costs_initial_ones.sort()
            costs_zeros.sort()
            
            min_moves_for_aliceIndex = float('inf')
            
            for num_changes in range(min(ones_needed, maxChanges) + 1):
                ones_from_initial = ones_needed - num_changes
                if ones_from_initial >= 0 and ones_from_initial <= initial_ones_count:
                    current_moves = 0
                    if num_changes > 0:
                        current_moves += sum(costs_zeros[:num_changes])
                    if ones_from_initial > 0:
                        current_moves += sum(costs_initial_ones[:ones_from_initial])
                    min_moves_for_aliceIndex = min(min_moves_for_aliceIndex, current_moves)
                    
            if min_moves_for_aliceIndex != float('inf'):
                min_total_moves = min(min_total_moves, min_moves_for_aliceIndex)
                
        return min_total_moves if min_total_moves != float('inf') else 0