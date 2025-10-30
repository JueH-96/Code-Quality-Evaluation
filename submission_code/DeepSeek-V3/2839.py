from bisect import bisect_left

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # Combine nums1 and nums2 into a list of tuples
        pairs = list(zip(nums1, nums2))
        # Sort the pairs based on nums1 in descending order
        pairs.sort(reverse=True, key=lambda x: x[0])
        
        # Preprocess queries to sort them based on x_i in descending order
        # and keep track of their original indices
        q_with_index = [(x, y, i) for i, (x, y) in enumerate(queries)]
        q_with_index.sort(reverse=True, key=lambda x: x[0])
        
        # Initialize the result list
        result = [-1] * len(queries)
        
        # Initialize a list to keep track of the maximum sums for each y
        # We will use a list of tuples (y, max_sum)
        max_sums = []
        
        # Iterate through the sorted queries and pairs
        idx = 0
        for x, y, q_idx in q_with_index:
            # Add all pairs where nums1[j] >= x to the max_sums list
            while idx < len(pairs) and pairs[idx][0] >= x:
                current_y, current_sum = pairs[idx][1], pairs[idx][0] + pairs[idx][1]
                # Maintain the max_sums list in a way that it is sorted by y
                # and for each y, we have the maximum sum
                # We can use a list and keep it sorted
                # To find the position to insert, we use bisect_left
                pos = bisect_left(max_sums, (current_y, -1))
                if pos < len(max_sums) and max_sums[pos][0] == current_y:
                    if max_sums[pos][1] < current_sum:
                        max_sums[pos] = (current_y, current_sum)
                else:
                    max_sums.insert(pos, (current_y, current_sum))
                idx += 1
            
            # Now, find the maximum sum where y_j >= y
            # We can use bisect_left to find the first y_j >= y
            pos = bisect_left(max_sums, (y, -1))
            if pos < len(max_sums):
                # Since max_sums is sorted by y, we need to find the maximum sum in the suffix
                # starting from pos
                max_sum = -1
                for i in range(pos, len(max_sums)):
                    if max_sums[i][1] > max_sum:
                        max_sum = max_sums[i][1]
                result[q_idx] = max_sum
            else:
                result[q_idx] = -1
        
        return result