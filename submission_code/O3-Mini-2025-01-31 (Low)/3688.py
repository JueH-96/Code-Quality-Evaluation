from typing import List

class Node:
    __slots__ = ('total', 'prefix', 'suffix', 'max_sum')
    def __init__(self, total, prefix, suffix, max_sum):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.max_sum = max_sum

def combine(left: Node, right: Node) -> Node:
    total = left.total + right.total
    prefix = max(left.prefix, left.total + right.prefix)
    suffix = max(right.suffix, right.total + left.suffix)
    max_sum = max(left.max_sum, right.max_sum, left.suffix + right.prefix)
    return Node(total, prefix, suffix, max_sum)

def build_tree(arr: List[int], tree: List[Node], idx: int, l: int, r: int):
    if l == r:
        val = arr[l]
        tree[idx] = Node(val, val, val, val)
        return 
    mid = (l + r) // 2
    build_tree(arr, tree, 2*idx, l, mid)
    build_tree(arr, tree, 2*idx+1, mid+1, r)
    tree[idx] = combine(tree[2*idx], tree[2*idx+1])

def query_tree(tree: List[Node], idx: int, l: int, r: int, ql: int, qr: int) -> Node:
    if ql <= l and r <= qr:
        return tree[idx]
    mid = (l + r) // 2
    if qr <= mid:
        return query_tree(tree, 2*idx, l, mid, ql, qr)
    elif ql > mid:
        return query_tree(tree, 2*idx+1, mid+1, r, ql, qr)
    else:
        left_node = query_tree(tree, 2*idx, l, mid, ql, qr)
        right_node = query_tree(tree, 2*idx+1, mid+1, r, ql, qr)
        return combine(left_node, right_node)

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        n = len(nums)
        # Helper: compute Kadane's maximum subarray sum for an array.
        def kadane(arr: List[int]) -> int:
            best = arr[0]
            cur = arr[0]
            for num in arr[1:]:
                cur = max(num, cur+num)
                best = max(best, cur)
            return best

        # Compute maximum subarray sum without removals.
        original_max = kadane(nums)
        
        # Build a segment tree over nums for fast range query (max subarray sum in any interval).
        tree = [None] * (4 * n)
        build_tree(nums, tree, 1, 0, n - 1)
        
        # Build a dictionary mapping candidate values to the list of their indices.
        # We only consider negative values since removing a positive will never help.
        positions = {}
        for i, num in enumerate(nums):
            if num < 0:
                if num not in positions:
                    positions[num] = []
                positions[num].append(i)
        
        best_after_removal = original_max  # initialize with original value, as "at most once" means we can choose not to remove
        # For every candidate negative value, consider removal.
        for x, idx_list in positions.items():
            # Now, removal of all x splits the array into segments.
            seg_max = float('-inf')
            start = 0
            for pos in idx_list:
                if start <= pos - 1:
                    # Query the segment [start, pos-1]
                    node = query_tree(tree, 1, 0, n - 1, start, pos - 1)
                    seg_max = max(seg_max, node.max_sum)
                start = pos + 1
            if start <= n - 1:
                node = query_tree(tree, 1, 0, n - 1, start, n - 1)
                seg_max = max(seg_max, node.max_sum)
            best_after_removal = max(best_after_removal, seg_max)
        
        return best_after_removal

# You can run the following test cases to verify your solution.
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1:
    nums = [-3, 2, -2, -1, 3, -2, 3]
    print(sol.maxSubarraySum(nums))  # Expected output: 7

    # Example 2:
    nums = [1,2,3,4]
    print(sol.maxSubarraySum(nums))  # Expected output: 10

    # Additional tests:
    nums = [-1, -2, -3, -4]  # all negatives
    print(sol.maxSubarraySum(nums))  # Expected: max possible single element if removal doesn't help