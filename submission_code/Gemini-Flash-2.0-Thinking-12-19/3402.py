from typing import List

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        min_val = min(nums)
        max_val = max(nums)
        min_cost = float('inf')
        for target_value in range(min_val, max_val + 3):
            required_increases = [max(0, target_value - num) for num in nums]
            current_cost = 0
            current_increases = list(required_increases)
            if cost2 >= 2 * cost1:
                current_cost = sum(current_increases) * cost1
            else:
                while True:
                    indices = []
                    for i in range(len(current_increases)):
                        if current_increases[i] > 0:
                            indices.append(i)
                    if len(indices) < 2:
                        break
                    index1 = indices[0]
                    index2 = -1
                    for idx in indices:
                        if idx != index1:
                            index2 = idx
                            break
                    if index2 == -1:
                        break
                    amount = min(current_increases[index1], current_increases[index2])
                    current_cost += amount * cost2
                    current_increases[index1] -= amount
                    current_increases[index2] -= amount
                current_cost += sum(current_increases) * cost1
            min_cost = min(min_cost, current_cost)
        return min_cost % (10**9 + 7)