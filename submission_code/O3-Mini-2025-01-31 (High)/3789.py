from typing import List
import math

# A Node represents the function F for an interval [l, r] (with l, r in 1-indexed coordinates)
# F(i) = RMQ(i, r) = (min_{j in [i, r]} X[j]).
# Since F is non-decreasing (as i increases the interval shrinks) the node stores a piecewise–constant function:
# "segs" is a list of tuples (s, e, value) meaning that for indices i with s <= i <= e the value is constant.
# Also store total = sum_{i=l}^r F(i), and min_val = F(l).
class Node:
    __slots__ = ('l', 'r', 'total', 'min_val', 'segs')
    def __init__(self, l: int, r: int, total: int, min_val: int, segs: List[tuple]):
        self.l = l
        self.r = r
        self.total = total
        self.min_val = min_val
        self.segs = segs

# For a leaf we have F(i)=X[i] (recall X[i] is defined – see below)
def make_leaf(idx: int, v: int) -> Node:
    # The piecewise function on just the single point is [(idx, idx, v)]
    return Node(idx, idx, v, v, [(idx, idx, v)])

# Merge two nodes: left covers [L, M] and right covers [M+1, R].
# For any i in [L,M] we have F(i) = min( left.F(i), right.min_val ).
# Then for i in [M+1, R] the function is given by right.segs.
def merge_nodes(left: Node, right: Node) -> Node:
    # (Assume both left and right are not None.)
    # Right child's overall minimum as a constant.
    B_min = right.min_val

    new_left_segs = []
    for (s, e, v) in left.segs:
        # For each segment in left, in the merged function the value is clamped to B_min:
        new_v = v if v < B_min else B_min
        if not new_left_segs:
            new_left_segs.append((s, e, new_v))
        else:
            # If the last segment has the same value, extend it.
            last_s, last_e, last_val = new_left_segs[-1]
            if last_val == new_v:
                new_left_segs[-1] = (last_s, e, new_v)
            else:
                new_left_segs.append((s, e, new_v))
    # Now our merged piece from left is given by new_left_segs.
    # We then “concatenate” right.segs.
    if new_left_segs and right.segs:
        # If the last segment of new_left_segs and the first of right.segs have the same value, merge them.
        if new_left_segs[-1][2] == right.segs[0][2]:
            s0, e0, val0 = new_left_segs[-1]
            s1, e1, val1 = right.segs[0]
            # Merge the two segments:
            new_left_segs[-1] = (s0, e1, val0)
            merged_segs = new_left_segs + right.segs[1:]
        else:
            merged_segs = new_left_segs + right.segs
    else:
        merged_segs = new_left_segs if new_left_segs else right.segs

    # Compute the total for the new node from left’s (modified) segments.
    left_total = 0
    for (s, e, val) in new_left_segs:
        left_total += (e - s + 1) * val
    new_total = left_total + right.total
    new_l = left.l
    new_r = right.r
    new_min_val = merged_segs[0][2]  # f(new_l)
    return Node(new_l, new_r, new_total, new_min_val, merged_segs)

# Identity element (for out–of–range leaves); here we simply use None in our iterative tree.
# When merging, if one child is None we return the other.
    
class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        # We first “build” an array X[1..n] where:
        # X[i] = min{ b such that [a,b] is a conflict pair with a=i } (if any) or n+1 if there is none.
        # Also, for a candidate conflict pair removal (we must remove exactly one),
        # the only “effective” ones are those where the smallest b appears uniquely.
        X = [n+1]*(n+1)   # 1-indexed; we will ignore index 0.
        # For each a that appears we also store the multiset of b's.
        conflict_dict = {}
        for pair in conflictingPairs:
            a, b = pair
            if a > b:
                a, b = b, a
            # note: conflict pair [a,b] forbids any subarray containing both a and b.
            # It shows up in X only at index a.
            if a not in conflict_dict:
                conflict_dict[a] = []
            conflict_dict[a].append(b)
        # candidate_info will store for each a (among those in conflict_dict)
        # a candidate new–value if we “remove” the unique minimum conflict pair.
        candidate_info = {}
        for a, Lb in conflict_dict.items():
            Lb.sort()
            m = Lb[0]
            # Count occurrences of m.
            cnt = 0
            for x in Lb:
                if x == m:
                    cnt += 1
                else:
                    break
            X[a] = m
            if cnt == 1:
                # The best we can do by “removing” the unique pair is to let X[a] become
                # the second–smallest b in Lb if it exists, or else n+1.
                if len(Lb) > 1:
                    candidate_info[a] = Lb[1]
                else:
                    candidate_info[a] = n+1
        # For i with no conflict, X[i] remains n+1.
        # Now the “number of valid subarrays” (after fixing the set of conflict–pairs)
        # is given by S – n(n+1)//2, where S = sum_{L=1}^n [ (min_{i in [L,n]} X[i]) ].
        # We now “build” a segment–tree which supports point–updates.
        # We'll use an iterative tree where the leaves (positions size ... size+n-1) correspond to indices 1..n.
        size = 1
        while size < n:
            size *= 2
        tree = [None]*(2*size)
        # Build the leaves.
        for i in range(1, n+1):
            tree[size + i - 1] = make_leaf(i, X[i])
        # For positions that are “dummy”, set to None.
        for i in range(n+1, size+1):
            tree[size + i - 1] = None
        # Merge bottom–up.
        for i in range(size-1, 0, -1):
            left = tree[2*i]
            right = tree[2*i+1]
            if left is None:
                tree[i] = right
            elif right is None:
                tree[i] = left
            else:
                tree[i] = merge_nodes(left, right)
        # The sum S is then tree[1].total
        f_original = tree[1].total - (n*(n+1)//2)
        best = f_original

        # Define a helper function to perform a point update at position pos (1-indexed) in X.
        def update(pos: int, new_val: int):
            index = pos - 1 + size
            tree[index] = make_leaf(pos, new_val)
            index //= 2
            while index:
                left = tree[2*index]
                right = tree[2*index+1]
                if left is None:
                    tree[index] = right
                elif right is None:
                    tree[index] = left
                else:
                    tree[index] = merge_nodes(left, right)
                index //= 2

        # For each candidate removal (only effective removals make a change),
        # we “temporarily” update X[a] to new candidate value.
        for a, new_val in candidate_info.items():
            old_val = X[a]
            if new_val == old_val:
                continue
            # (Do a point update and query the overall total)
            update(a, new_val)
            cand_total = tree[1].total
            candidate_f = cand_total - (n*(n+1)//2)
            if candidate_f > best:
                best = candidate_f
            # Revert update.
            update(a, old_val)
        return best