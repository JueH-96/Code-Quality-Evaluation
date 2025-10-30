from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        """
        We want to pick 2*k elements as a subsequence (in order) and split them into two
        groups of k each. The value is (OR of first k) XOR (OR of last k). We seek the maximum
        such value.

        Approach:
          1) Let n = len(nums). We'll build two DP tables (dpL, dpR), each of size (n+1) x (k+1),
             where each entry dp[..][..] is a 128-bit mask (since nums[i] < 128) representing
             all possible OR-values obtainable by picking some subsequence of length r in
             the specified subarray.

             - dpL[i][r]: bitmask of all possible OR-values of length-r subsequences using
               nums[0..i-1].
             - dpR[i][r]: bitmask of all possible OR-values of length-r subsequences using
               nums[i..n-1].

          2) Fill dpL left-to-right:
               dpL[0][0] = (1 << 0)  # means OR=0 is possible with an empty subsequence
               dpL[0][r>0] = 0
               Then for i in [0..n-1]:
                 for r in [0..k]:
                   - Skip nums[i]: 
                     dpL[i+1][r] |= dpL[i][r]
                   - Take nums[i] (if r < k):
                     for each OR-value 'val' in dpL[i][r], we can form new OR = (val | nums[i])
                     set that bit in dpL[i+1][r+1].

          3) Fill dpR right-to-left:
               dpR[n][0] = (1 << 0)
               dpR[n][r>0] = 0
               Then for i in [n-1..0]:
                 for r in [0..k]:
                   - Skip nums[i]:
                     dpR[i][r] |= dpR[i+1][r]
                   - Take nums[i] (if r < k):
                     for each OR-value 'val' in dpR[i+1][r], form (val | nums[i]).

          4) The subsequence is split so that the first k elements come from [0..j-1],
             and the second k elements come from [j..n-1], with no overlap. Hence we combine:
               - Lmask = dpL[j][k]
               - Rmask = dpR[j][k]
             for j in [k..n-k], ensuring at least k elements in each part.

          5) To combine two sets of possible OR-values and find max XOR, we build a small
             7-bit trie for Rmask, and then query each value in Lmask for its best XOR.

          This DP + bit-trie approach runs within the constraints (n <= 400, k <= 200, nums[i] < 128).
        """

        n = len(nums)

        # Precompute a helper to OR-transform a bitmask with a single number x
        # mask has bits set at indices = OR-values
        # We want to set the bit at (old_val | x) for each old_val in mask
        def or_transform(mask: int, x: int) -> int:
            res = 0
            curr = mask
            while curr:
                b = curr & -curr  # lowest set bit
                idx = b.bit_length() - 1
                new_val = idx | x
                res |= 1 << new_val
                curr ^= b
            return res

        # Build dpL: dpL[i][r] is a bitmask of all possible OR-values using
        # up to the i-th element (nums[:i]) and picking exactly r elements.
        dpL = [[0]*(k+1) for _ in range(n+1)]
        dpL[0][0] = 1 << 0  # OR=0 with empty subset
        for i in range(n):
            val = nums[i]
            for r in range(k+1):
                # Skip nums[i]
                dpL[i+1][r] |= dpL[i][r]
                # Take nums[i], if r < k
                if r < k and dpL[i][r] != 0:
                    dpL[i+1][r+1] |= or_transform(dpL[i][r], val)

        # Build dpR: dpR[i][r] is a bitmask of all possible OR-values using
        # nums[i:] (elements from i to end) with exactly r elements.
        dpR = [[0]*(k+1) for _ in range(n+1)]
        dpR[n][0] = 1 << 0
        for i in range(n-1, -1, -1):
            val = nums[i]
            for r in range(k+1):
                # Skip nums[i]
                dpR[i][r] |= dpR[i+1][r]
                # Take nums[i], if r < k
                if r < k and dpR[i+1][r] != 0:
                    dpR[i][r+1] |= or_transform(dpR[i+1][r], val)

        # Build a small 7-bit binary trie to speed up max-XOR queries.
        class TrieNode:
            __slots__ = ['child']
            def __init__(self):
                # child[0], child[1]
                self.child = [None, None]

        def trie_insert(root: TrieNode, val: int):
            node = root
            # We only need 7 bits (since val < 128)
            for bit in reversed(range(7)):
                b = (val >> bit) & 1
                if node.child[b] is None:
                    node.child[b] = TrieNode()
                node = node.child[b]

        def trie_query(root: TrieNode, val: int) -> int:
            """Return the maximum XOR achievable by val with some number in the trie."""
            node = root
            ans = 0
            for bit in reversed(range(7)):
                b = (val >> bit) & 1
                toggle = 1 - b
                if node.child[toggle] is not None:
                    ans |= (1 << bit)
                    node = node.child[toggle]
                else:
                    node = node.child[b]
            return ans

        def build_trie_from_mask(mask: int) -> TrieNode:
            root = TrieNode()
            curr = mask
            while curr:
                lowest = curr & -curr
                idx = lowest.bit_length() - 1
                trie_insert(root, idx)
                curr ^= lowest
            return root

        best_answer = 0
        # Try splitting at j in [k..(n-k)], so that the first k come from dpL[j]
        # and the second k come from dpR[j].
        for j in range(k, n - k + 1):
            left_mask = dpL[j][k]
            right_mask = dpR[j][k]
            if left_mask == 0 or right_mask == 0:
                # No valid subsequence of length k at this boundary
                continue

            # Build trie for right half
            right_trie = build_trie_from_mask(right_mask)

            # For each OR-value in left half, query the trie for max XOR
            curr = left_mask
            while curr:
                b = curr & -curr
                val = b.bit_length() - 1
                curr ^= b
                # Query
                candidate = trie_query(right_trie, val)
                if candidate > best_answer:
                    best_answer = candidate

        return best_answer