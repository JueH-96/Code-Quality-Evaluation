import collections
import math
from typing import List

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        max_val = max(nums)

        # 1. Count frequencies of numbers in nums
        # Use a list for easier indexing up to max_val
        freq_list = [0] * (max_val + 1)
        for num in nums:
            freq_list[num] += 1

        # 2. Count how many numbers in nums are divisible by v, for v = 1 to max_val
        # count_divisible_arr[v] = number of nums[i] such that v | nums[i]
        count_divisible_arr = [0] * (max_val + 1)
        # Iterate through each possible divisor v
        for v in range(1, max_val + 1):
            # Iterate through multiples m of v up to max_val
            # Sum freq_list[m] for all multiples m of v
            for m in range(v, max_val + 1, v):
                count_divisible_arr[v] += freq_list[m]

        # 3. Count how many pairs (i, j) with i < j have both nums[i] and nums[j] divisible by v
        # This is the number of pairs whose GCD is a multiple of v
        # CountMultiple_arr[v] = Count({ (i, j) | i < j, v | nums[i] and v | nums[j] })
        # If there are c numbers divisible by v, there are c*(c-1)/2 pairs.
        CountMultiple_arr = [0] * (max_val + 1)
        for v in range(1, max_val + 1):
            c = count_divisible_arr[v]
            CountMultiple_arr[v] = c * (c - 1) // 2

        # 4. Count how many pairs (i, j) with i < j have gcd(nums[i], nums[j]) = v
        # C_exactly_arr[v] = Count({ (i, j) | i < j, gcd(nums[i], nums[j]) = v })
        # We know CountMultiple[v] = sum_{k=1}^{max_val//v} C_exactly[k*v]
        # We can calculate C_exactly by iterating v from max_val down to 1
        # C_exactly[v] = CountMultiple[v] - sum_{k=2}^{max_val//v} C_exactly[k*v]
        C_exactly_arr = [0] * (max_val + 1)
        for v in range(max_val, 0, -1):
            C_exactly_arr[v] = CountMultiple_arr[v]
            # Subtract counts of pairs whose GCD is a multiple of v > v
            # Iterate through multiples k*v where k >= 2
            # The loop condition `k*v <= max_val` is implicitly handled by `k in range(2, max_val // v + 1)`
            # since `k <= max_val // v` implies `k * v <= max_val`.
            for k in range(2, max_val // v + 1):
                 C_exactly_arr[v] -= C_exactly_arr[k * v]

        # 5. Compute cumulative counts
        # cumulative_count[v] = number of pairs with GCD <= v
        cumulative_count = [0] * (max_val + 1)
        for v in range(1, max_val + 1):
            cumulative_count[v] = cumulative_count[v - 1] + C_exactly_arr[v]

        # 6. Answer queries using binary search on cumulative counts
        answer = []
        for query_k in queries:
            # We need the (query_k + 1)-th smallest GCD (since query_k is 0-indexed)
            target_count = query_k + 1

            # Find the smallest g in [1, max_val] such that cumulative_count[g] >= target_count
            # Binary search finds the first index `g` where cumulative_count[g] is >= target_count
            low = 1
            high = max_val
            ans_g = max_val # Initialize with a possible answer (upper bound)

            while low <= high:
                mid = (low + high) // 2
                if cumulative_count[mid] >= target_count:
                    # mid is a possible answer or the answer is smaller
                    ans_g = mid
                    high = mid - 1
                else:
                    # mid is too small, the answer must be larger
                    low = mid + 1
            answer.append(ans_g)

        return answer