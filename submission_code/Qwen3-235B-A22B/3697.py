from typing import List
from math import gcd
from itertools import product

class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        def compute_lcm(arr):
            current_lcm = 1
            for num in arr:
                current_lcm = current_lcm * num // gcd(current_lcm, num)
            return current_lcm

        def generate_partitions(s):
            if not s:
                yield []
                return
            last = s[-1]
            rest = s[:-1]
            for partition in generate_partitions(rest):
                # Add 'last' to each of the existing subsets in the partition
                for i in range(len(partition)):
                    new_part = [subset.copy() for subset in partition]
                    new_part[i].append(last)
                    yield new_part
                # Add 'last' as a new subset
                new_part = [subset.copy() for subset in partition]
                new_part.append([last])
                yield new_part

        if not target:
            return 0  # As per constraints, target is non-empty

        all_partitions = list(generate_partitions(target))
        min_total = float('inf')

        for partition in all_partitions:
            S = len(partition)
            if S > len(nums):
                continue

            lcm_list = [compute_lcm(subset) for subset in partition]

            subset_candidate_lists = []
            for lcm_val in lcm_list:
                candidates = []
                for i in range(len(nums)):
                    num = nums[i]
                    q = (num + lcm_val - 1) // lcm_val
                    next_multiple = q * lcm_val
                    cost = next_multiple - num
                    candidates.append((cost, num, i))
                candidates.sort(key=lambda x: (x[0], x[2]))
                subset_candidate_lists.append(candidates[:S])

            ranges = [range(S) for _ in range(S)]
            for indices in product(*ranges):
                used_indices = set()
                total_cost = 0
                valid = True
                for subset_idx in range(S):
                    candidate_idx = indices[subset_idx]
                    cost, _, num_idx = subset_candidate_lists[subset_idx][candidate_idx]
                    if num_idx in used_indices:
                        valid = False
                        break
                    used_indices.add(num_idx)
                    total_cost += cost
                if valid and total_cost < min_total:
                    min_total = total_cost

        return min_total