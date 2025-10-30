from typing import List

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        # Total number of subarrays
        T = n * (n + 1) // 2

        # (A) Compute sum of distinct counts over all subarrays.
        # For each index i, the number of subarrays where nums[i] is the 
        # FIRST occurrence of that value is (i - prev_index) * (n - i)
        # Summing these up gives the total “distinct count” summed over subarrays.
        sum_distinct = 0
        last_occ = {}
        for i, num in enumerate(nums):
            prev = last_occ.get(num, -1)
            sum_distinct += (i - prev) * (n - i)
            last_occ[num] = i

        # (B) The imbalance of a subarray (after sorting its unique values)
        # is defined as the number of adjacent “gaps” where the difference is > 1.
        # If S is the set of distinct values in the subarray sorted in increasing order,
        # then observe that if S is contiguous, it has |S|-1 consecutive pairs 
        # (every adjacent difference is 1) and imbalance is 0.
        # Otherwise, the number of adjacent pairs that are NOT consecutive is:
        #      imbalance = (|S| - 1) - (# of pairs of the form (x, x+1) that occur in S).
        #
        # Summing this over all subarrays gives:
        #      ∑[imbalance] = ∑[|S|] - (# subarrays) - ∑[# of pairs (x, x+1) in S].
        #
        # We already computed ∑[|S|] as sum_distinct and we know (# subarrays) = T.
        # So it remains to compute S_pairs = ∑[# of pairs (x,x+1) in S]
        # Notice that for any fixed x, the distinct set S includes the pair (x, x+1)
        # exactly when the subarray contains at least one occurrence of x and at least one occurrence of x+1.
        # Hence, if we can count for each x (with x in [1, n-1] since nums[i] ∈ [1,n])
        # the total number of subarrays that contain both x and x+1,
        # then the sum over x is exactly ∑[# of consecutive pairs in S] over all subarrays.
        
        # To help with that we’ll first record, for each value in [1, n], the sorted list of indices where it occurs.
        occ = [[] for _ in range(n + 1)]
        for i, num in enumerate(nums):
            occ[num].append(i)
            
        # Given a sorted list of indices where a value occurs,
        # count the number of subarrays that do NOT include ANY of these indices.
        # A subarray fails to include the value if it is completely contained in one of the “gaps”
        # between indices where the value appears.
        def count_missing(indices: List[int], n: int) -> int:
            total_missing = 0
            prev_index = -1
            # For gap before the first occurrence and between occurrences.
            for idx in indices:
                gap = idx - prev_index - 1
                total_missing += gap * (gap + 1) // 2
                prev_index = idx
            # Gap after the last occurrence.
            gap = n - prev_index - 1
            total_missing += gap * (gap + 1) // 2
            return total_missing

        # A simple merge of two sorted lists.
        def merge_sorted(a: List[int], b: List[int]) -> List[int]:
            merged = []
            i, j = 0, 0
            while i < len(a) and j < len(b):
                if a[i] < b[j]:
                    merged.append(a[i])
                    i += 1
                else:
                    merged.append(b[j])
                    j += 1
            while i < len(a):
                merged.append(a[i])
                i += 1
            while j < len(b):
                merged.append(b[j])
                j += 1
            return merged

        # For every possible consecutive pair (x, x+1) (with x from 1 to n-1)
        # count the number of subarrays that contain both x and x+1.
        # Using inclusion-exclusion:
        #    count(both) = T - missing(x) - missing(x+1) + missing(merged(x, x+1))
        # where missing(v) is the number of subarrays that do NOT contain v.
        sum_pairs = 0
        for x in range(1, n):  # For x in 1 ... n-1 (so that x+1 is ≤ n)
            if occ[x] and occ[x+1]:
                missing_x = count_missing(occ[x], n)
                missing_xp1 = count_missing(occ[x+1], n)
                merged_list = merge_sorted(occ[x], occ[x+1])
                missing_merged = count_missing(merged_list, n)
                # Count of subarrays that contain both x and x+1.
                count_both = T - missing_x - missing_xp1 + missing_merged
                sum_pairs += count_both

        # (C) Finally, putting it together:
        # Total imbalance sum = (sum of distinct counts) - (number of subarrays) - (sum over subarrays of consecutive pairs)
        return sum_distinct - T - sum_pairs