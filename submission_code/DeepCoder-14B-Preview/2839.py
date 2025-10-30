import bisect

class Solution:
    def maximumSumQueries(self, nums1: list, nums2: list, queries: list) -> list:
        # Combine the nums into a list of tuples, sorted by a in ascending order
        combined = sorted(zip(nums1, nums2), key=lambda x: x[0])
        a_list = [x[0] for x in combined]
        n = len(combined)
        
        # Build the segment tree
        class SegmentTreeNode:
            def __init__(self, l, r):
                self.l = l
                self.r = r
                self.left = None
                self.right = None
                self.data = []  # list of (b, sum) tuples, sorted by b
                self.suffix_max = []  # suffix maximum array
        
        def build(l, r):
            node = SegmentTreeNode(l, r)
            if l == r:
                b = combined[l][1]
                s = combined[l][0] + b
                node.data = [(b, s)]
                # Compute suffix_max
                max_sum = -float('inf')
                suffix_max = []
                for i in reversed(range(len(node.data))):
                    current_sum = node.data[i][1]
                    if current_sum > max_sum:
                        max_sum = current_sum
                    suffix_max.append(max_sum)
                suffix_max.reverse()
                node.suffix_max = suffix_max
            else:
                mid = (l + r) // 2
                node.left = build(l, mid)
                node.right = build(mid + 1, r)
                # Merge the two sorted lists
                i = j = 0
                left_data = node.left.data
                right_data = node.right.data
                merged = []
                while i < len(left_data) and j < len(right_data):
                    if left_data[i][0] < right_data[j][0]:
                        merged.append(left_data[i])
                        i += 1
                    else:
                        merged.append(right_data[j])
                        j += 1
                merged += left_data[i:]
                merged += right_data[j:]
                node.data = merged
                # Compute suffix_max
                max_sum = -float('inf')
                suffix_max = []
                for i in reversed(range(len(node.data))):
                    current_sum = node.data[i][1]
                    if current_sum > max_sum:
                        max_sum = current_sum
                    suffix_max.append(max_sum)
                suffix_max.reverse()
                node.suffix_max = suffix_max
            return node
        
        if n == 0:
            return []
        root = build(0, n - 1)
        
        # Function to query the segment tree
        def query_segment(node, l, r, y):
            if node.r < l or node.l > r:
                return -float('inf')
            if l <= node.l and node.r <= r:
                # Binary search for y in node.data
                # node.data is sorted by b
                left = 0
                right = len(node.data)
                while left < right:
                    mid = (left + right) // 2
                    if node.data[mid][0] < y:
                        left = mid + 1
                    else:
                        right = mid
                if left >= len(node.data):
                    return -float('inf')
                # The maximum sum from left to end
                if left >= len(node.suffix_max):
                    return -float('inf')
                return node.suffix_max[left]
            else:
                left_max = query_segment(node.left, l, r, y) if node.left else -float('inf')
                right_max = query_segment(node.right, l, r, y) if node.right else -float('inf')
                return max(left_max, right_max)
        
        # Process each query
        result = []
        for q in queries:
            x, y = q[0], q[1]
            # Find the first index where a >= x
            i = bisect.bisect_left(a_list, x)
            if i >= n:
                result.append(-1)
                continue
            # Query the segment tree from i to n-1
            max_sum = query_segment(root, i, n - 1, y)
            if max_sum == -float('inf'):
                result.append(-1)
            else:
                result.append(max_sum)
        return result