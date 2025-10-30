from bisect import bisect_left

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # Combine nums1 and nums2 into a list of tuples
        pairs = list(zip(nums1, nums2))
        # Sort the pairs based on nums1 in descending order
        pairs.sort(reverse=True, key=lambda x: x[0])
        
        # Preprocess the queries to sort them based on x_i in descending order
        # and keep track of their original indices
        indexed_queries = [(x, y, i) for i, (x, y) in enumerate(queries)]
        indexed_queries.sort(reverse=True, key=lambda x: x[0])
        
        # Initialize the result array
        result = [-1] * len(queries)
        
        # Initialize a list to keep track of the maximum sums for each y_i
        # We will maintain a list of (y_i, max_sum) pairs, sorted by y_i
        max_sums = []
        
        # Iterate through the sorted queries and pairs
        pair_ptr = 0
        for x, y, idx in indexed_queries:
            # Add all pairs where nums1[j] >= x to the max_sums list
            while pair_ptr < len(pairs) and pairs[pair_ptr][0] >= x:
                current_y, current_sum = pairs[pair_ptr][1], pairs[pair_ptr][0] + pairs[pair_ptr][1]
                # Update the max_sums list
                # We need to maintain the list in a way that for each y_i, we have the maximum sum
                # So we can use a list of tuples (y_i, max_sum) and keep it sorted by y_i
                # When we add a new pair, we need to find the position where it should be inserted
                # and update the max_sum accordingly
                # To optimize, we can use a list that is sorted by y_i and for each y_i, we keep the maximum sum
                # So we can perform a binary search to find the position to insert the new y_i
                # and update the max_sum if necessary
                # Here, we will use a list of tuples (y_i, max_sum) and keep it sorted by y_i
                # We will perform a binary search to find the position to insert the new y_i
                # and update the max_sum if necessary
                # First, find the position where the new y_i should be inserted
                pos = bisect_left(max_sums, (current_y, -1))
                if pos < len(max_sums) and max_sums[pos][0] == current_y:
                    # If the y_i already exists, update the max_sum if the new sum is greater
                    if current_sum > max_sums[pos][1]:
                        max_sums[pos] = (current_y, current_sum)
                else:
                    # Insert the new y_i and max_sum
                    max_sums.insert(pos, (current_y, current_sum))
                # Now, we need to ensure that for all y_i >= current_y, the max_sum is at least current_sum
                # So we can iterate from the current position to the end and update the max_sum if necessary
                i = pos + 1
                while i < len(max_sums):
                    if max_sums[i][1] < current_sum:
                        max_sums[i] = (max_sums[i][0], current_sum)
                    i += 1
                pair_ptr += 1
            # Now, find the maximum sum for the current query
            # We need to find the maximum sum among all y_i >= y
            # Since max_sums is sorted by y_i, we can perform a binary search to find the first y_i >= y
            pos = bisect_left(max_sums, (y, -1))
            if pos < len(max_sums):
                result[idx] = max_sums[pos][1]
        return result