from typing import List

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        """
        Finds the minimum absolute difference between two elements at least x indices apart.

        The solution iterates through the array, maintaining a sorted collection of 
        elements seen so far (from the 'left' part of the array, `x` indices away). 
        For each new element, it finds the closest values (predecessor and successor)
        in the sorted collection to calculate the minimum possible difference.

        To achieve this efficiently, a Fenwick tree (Binary Indexed Tree) is used on 
        coordinate-compressed values. This allows for adding elements and finding the k-th
        element (and thus predecessors/successors) in O(log n) time.
        """
        if x == 0:
            return 0

        n = len(nums)
        
        # Coordinate Compression: Map values to ranks 0 to m-1
        unique_sorted_nums = sorted(list(set(nums)))
        m = len(unique_sorted_nums)
        val_to_rank = {val: i for i, val in enumerate(unique_sorted_nums)}

        # Fenwick Tree (BIT) for maintaining counts of ranks
        bit = [0] * (m + 1)

        def update(i, delta):
            i += 1
            while i <= m:
                bit[i] += delta
                i += i & (-i)

        def query(i):
            if i < 0:
                return 0
            i += 1
            s = 0
            while i > 0:
                s += bit[i]
                i -= i & (-i)
            return s

        def find_kth(k):
            # Finds the rank of the k-th element in the BIT
            if k <= 0: return -1
            curr, p = 0, 1 << (m.bit_length() - 1)
            while p > 0:
                if curr + p <= m and bit[curr + p] < k:
                    k -= bit[curr + p]
                    curr += p
                p >>= 1
            return curr if curr < m else -1

        min_diff = float('inf')
        total_elements = 0

        # Iterate through the array, maintaining elements from nums[0...j-x] in the BIT
        for j in range(x, n):
            # Add nums[j-x] to the set of considered elements
            rank_to_add = val_to_rank[nums[j-x]]
            update(rank_to_add, 1)
            total_elements += 1

            current_val = nums[j]
            current_rank = val_to_rank[current_val]

            # Find predecessor: the largest element in the set <= current_val
            count_le = query(current_rank)
            if count_le > 0:
                pred_rank = find_kth(count_le)
                if pred_rank != -1:
                    pred_val = unique_sorted_nums[pred_rank]
                    min_diff = min(min_diff, current_val - pred_val)

            # Find successor: the smallest element in the set >= current_val
            count_lt = query(current_rank - 1)
            if count_lt < total_elements:
                succ_rank = find_kth(count_lt + 1)
                if succ_rank != -1:
                    succ_val = unique_sorted_nums[succ_rank]
                    min_diff = min(min_diff, succ_val - current_val)

            if min_diff == 0:
                return 0
                
        return min_diff