class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        # Initialize the maximum subarray XOR scores list
        max_xor_scores = [0] * len(queries)

        # Create a prefix XOR array
        prefix_xor = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            prefix_xor[i + 1] = prefix_xor[i] ^ nums[i]

        # Create a segment tree for range maximum queries
        segment_tree = [0] * (4 * len(prefix_xor))

        def build_tree(node, start, end):
            if start == end:
                segment_tree[node] = prefix_xor[start]
            else:
                mid = (start + end) // 2
                build_tree(2 * node + 1, start, mid)
                build_tree(2 * node + 2, mid + 1, end)
                segment_tree[node] = max(segment_tree[2 * node + 1], segment_tree[2 * node + 2])

        build_tree(0, 0, len(prefix_xor) - 1)

        def query_tree(node, start, end, left, right):
            if start > right or end < left:
                return 0
            elif start >= left and end <= right:
                return segment_tree[node]
            else:
                mid = (start + end) // 2
                return max(query_tree(2 * node + 1, start, mid, left, right),
                           query_tree(2 * node + 2, mid + 1, end, left, right))

        for i, (l, r) in enumerate(queries):
            max_xor_scores[i] = query_tree(0, 0, len(prefix_xor) - 1, l, r) ^ prefix_xor[l]

        return max_xor_scores