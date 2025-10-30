class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        from bisect import bisect_left
        from collections import defaultdict
        
        # Sort coordinates by x first, then by y
        coordinates.sort()
        
        # Extract the point of interest
        start_point = coordinates[k]
        
        # Create a list of y-values for points with x-values less than or equal to start_point's x
        y_values = [y for x, y in coordinates if x <= start_point[0]]
        
        # Function to find the length of the longest increasing subsequence
        def longest_increasing_subsequence(seq):
            lis = []
            for num in seq:
                pos = bisect_left(lis, num)
                if pos == len(lis):
                    lis.append(num)
                else:
                    lis[pos] = num
            return len(lis)
        
        # Find the longest increasing subsequence of y-values
        return longest_increasing_subsequence(y_values)