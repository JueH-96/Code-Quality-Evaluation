from typing import List

class Solution:
    class Node:
        __slots__ = ('sum', 'pref', 'suff', 'best')
        def __init__(self, s, p, su, b):
            self.sum = s
            self.pref = p
            self.suff = su
            self.best = b

    def merge(self, left: 'Solution.Node', right: 'Solution.Node') -> 'Solution.Node':
        # Merge two segment-tree nodes
        s = left.sum + right.sum
        p = left.pref if left.pref >= left.sum + right.pref else left.sum + right.pref
        su = right.suff if right.suff >= right.sum + left.suff else right.sum + left.suff
        # best is max of left.best, right.best, left.suff + right.pref
        cross = left.suff + right.pref
        b = left.best if left.best >= right.best else right.best
        if cross > b:
            b = cross
        return Solution.Node(s, p, su, b)

    def build_tree(self, nums: List[int], tree: List['Solution.Node'], idx: int, l: int, r: int):
        if l == r:
            v = nums[l]
            # leaf node
            tree[idx] = Solution.Node(v, v, v, v)
        else:
            mid = (l + r) // 2
            left = idx * 2
            right = left + 1
            self.build_tree(nums, tree, left, l, mid)
            self.build_tree(nums, tree, right, mid+1, r)
            tree[idx] = self.merge(tree[left], tree[right])

    def query_tree(self, tree: List['Solution.Node'], idx: int, l: int, r: int, ql: int, qr: int) -> 'Solution.Node':
        # query the node info for interval [ql, qr]
        if ql <= l and r <= qr:
            return tree[idx]
        mid = (l + r) // 2
        res = None
        if ql <= mid:
            left_node = self.query_tree(tree, idx*2, l, mid, ql, qr)
            res = left_node
        if qr > mid:
            right_node = self.query_tree(tree, idx*2+1, mid+1, r, ql, qr)
            if res is None:
                res = right_node
            else:
                res = self.merge(res, right_node)
        return res

    def maxSubarraySum(self, nums: List[int]) -> int:
        n = len(nums)
        # Build segment tree
        # tree size: 4*n is safe
        tree = [None] * (4 * n)
        self.build_tree(nums, tree, 1, 0, n-1)
        # original max subarray (no deletion) is the best at root
        ans = tree[1].best

        # map each value to list of positions
        pos = {}
        for i, v in enumerate(nums):
            pos.setdefault(v, []).append(i)

        # for each candidate x, remove all x and compute max subarray via merging intervals
        for x, lst in pos.items():
            # skipping x if removing all x leaves empty array
            if len(lst) == n:
                continue
            # build intervals of [l, r] segments where nums[i] != x
            intervals = []
            # before first occurrence
            first = lst[0]
            if first > 0:
                intervals.append((0, first - 1))
            # between occurrences
            for j in range(1, len(lst)):
                prev = lst[j-1]
                cur = lst[j]
                if prev + 1 <= cur - 1:
                    intervals.append((prev+1, cur-1))
            # after last
            last = lst[-1]
            if last < n-1:
                intervals.append((last+1, n-1))
            if not intervals:
                # no segments remain
                continue
            # merge segment-tree nodes for these intervals in order
            l0, r0 = intervals[0]
            node = self.query_tree(tree, 1, 0, n-1, l0, r0)
            # merge subsequent intervals
            for (l1, r1) in intervals[1:]:
                nd = self.query_tree(tree, 1, 0, n-1, l1, r1)
                node = self.merge(node, nd)
            # update answer
            if node.best > ans:
                ans = node.best

        return ans