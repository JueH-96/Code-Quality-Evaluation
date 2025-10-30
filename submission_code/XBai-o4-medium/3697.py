import math
from math import gcd
from typing import List
from itertools import product

class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        # Function to generate all set partitions of the target list
        def generate_partitions(collection):
            if not collection:
                yield []
                return
            first = collection[0]
            for smaller in generate_partitions(collection[1:]):
                for i in range(len(smaller)):
                    new_smaller = smaller.copy()
                    new_smaller[i] = [first] + new_smaller[i]
                    yield new_smaller
                yield [[first]] + smaller

        # Function to compute LCM of a list of numbers
        def list_lcm(numbers):
            def lcm(a, b):
                return a * b // gcd(a, b)
            current_lcm = numbers[0]
            for num in numbers[1:]:
                current_lcm = lcm(current_lcm, num)
            return current_lcm

        # Generate all possible partitions of the target list
        all_partitions = []
        for p in generate_partitions(target):
            all_partitions.append(p)

        min_total = float('inf')
        for partition in all_partitions:
            subsets_candidates = []
            for subset in partition:
                L = list_lcm(subset)
                candidates = []
                for x in nums:
                    # Calculate the smallest multiple of L >= x
                    multiple = ((x + L - 1) // L) * L
                    cost = multiple - x
                    candidates.append((cost, x))
                # Sort candidates by cost and take top K candidates
                candidates.sort()
                top_k = len(partition)
                top_candidates = candidates[:top_k]
                subsets_candidates.append(top_candidates)
            # Check if all subsets have at least one candidate
            valid = True
            for sc in subsets_candidates:
                if not sc:
                    valid = False
                    break
            if not valid:
                continue
            # Generate all possible combinations of candidates
            candidate_lists = subsets_candidates
            m = len(candidate_lists)
            for combo in product(*candidate_lists):
                elements = [x for (cost, x) in combo]
                if len(elements) == len(set(elements)):
                    total_cost = sum(cost for (cost, x) in combo)
                    if total_cost < min_total:
                        min_total = total_cost
        return min_total