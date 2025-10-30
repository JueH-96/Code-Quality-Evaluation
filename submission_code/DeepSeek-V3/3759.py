import bisect

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        elements = []
        for i in range(n):
            elements.append((nums1[i], nums2[i], i))
        # Sort elements based on nums1 in ascending order
        elements.sort()
        
        # Coordinate compression for nums2 values
        unique_nums2 = sorted(set(nums2))
        rank_map = {v: i+1 for i, v in enumerate(unique_nums2)}
        max_rank = len(unique_nums2)
        
        # Binary Indexed Tree (Fenwick Tree) to maintain top k elements
        # We need two BITs: one for count and one for sum
        class FenwickTree:
            def __init__(self, size):
                self.size = size
                self.count_tree = [0] * (self.size + 2)
                self.sum_tree = [0] * (self.size + 2)
            
            def update(self, rank, delta_count, delta_sum):
                while rank <= self.size:
                    self.count_tree[rank] += delta_count
                    self.sum_tree[rank] += delta_sum
                    rank += rank & -rank
            
            def query_k_largest(self, k):
                res_sum = 0
                res_count = 0
                rank = self.size
                while rank > 0:
                    if res_count + self.count_tree[rank] <= k:
                        res_count += self.count_tree[rank]
                        res_sum += self.sum_tree[rank]
                        rank -= rank & -rank
                    else:
                        # Need to find the exact position within this node
                        # Binary search within the current node's children (not straightforward)
                        # Alternative: step down bit by bit
                        break
                if res_count < k and rank > 0:
                    # Now, we need to find the largest rank such that the cumulative count is <=k
                    # Since BIT is not a perfect binary search tree, we proceed bit by bit
                    temp_rank = rank
                    step = 1 << (temp_rank.bit_length() - 1)
                    while step > 0:
                        next_rank = temp_rank - step
                        if next_rank >= 0:
                            if res_count + self.count_tree[next_rank] <= k:
                                res_count += self.count_tree[next_rank]
                                res_sum += self.sum_tree[next_rank]
                                temp_rank = next_rank
                        step >>= 1
                    # The remaining elements are in temp_rank's position
                    if temp_rank > 0 and res_count < k:
                        cnt_in_node = self.count_tree[temp_rank]
                        sum_in_node = self.sum_tree[temp_rank]
                        take = k - res_count
                        # The value is unique_nums2[temp_rank-1], but how many are there?
                        # Since all in this node have the same value (if they are leaves)
                        # So each element in this node has value unique_nums2[temp_rank-1]
                        res_sum += take * unique_nums2[temp_rank - 1]
                        res_count += take
                return res_sum
        
        ft = FenwickTree(max_rank)
        
        # Process elements in nums1 order, but answer queries in original order
        # We need to process elements in nums1 ascending order, and for each i, the answer is the sum of top k elements in nums2 where nums1[j] < nums1[i], processed before i.
        # So we need to process elements in sorted order, and for each element, after processing, we can answer queries for elements larger than current nums1.
        # But the original problem requires for each i, the answer is computed based on all j where nums1[j] < nums1[i], regardless of their position.
        # So we can process all elements in sorted order, and for each element, after inserting into the BIT, we can store the answer for all elements with nums1[i] > current nums1.
        # However, the original i's are in arbitrary order. So we need to collect all queries, process them in sorted order, and then assign the answers back.
        
        # We'll create a list of queries, sorted by nums1[i], and process them in order, using the BIT to keep track of the top k elements up to that point.
        queries = []
        for i in range(n):
            queries.append((nums1[i], i))
        queries.sort()
        
        res = [0] * n
        ptr = 0
        # Process queries in order of increasing nums1[i]
        for num, i in queries:
            # Process all elements in 'elements' with nums1[j] < num, not yet processed
            while ptr < n and elements[ptr][0] < num:
                val1, val2, original_pos = elements[ptr]
                rank = rank_map[val2]
                ft.update(rank, 1, val2)
                ptr += 1
            # Now, query the top k elements in the BIT
            if k == 0:
                res[i] = 0
            else:
                sum_k = ft.query_k_largest(k)
                res[i] = sum_k
        return res