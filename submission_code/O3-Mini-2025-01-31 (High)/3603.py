from typing import List
import sys

class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        # In worst-case (chain tree) the recursive dfs may be deep.
        sys.setrecursionlimit(10**6)
        n = len(parent)
        
        # Build the children list for each node.
        children = [[] for _ in range(n)]
        for i in range(1, n):
            children[parent[i]].append(i)
        # Ensure that the children are processed in increasing order.
        for child_list in children:
            child_list.sort()
        
        # We will use a polynomial rolling hash.
        mod = 10**9 + 7
        base = 31
        # Precompute powers of base up to n (since maximum length <= n).
        pow_base = [1] * (n + 1)
        for i in range(1, n + 1):
            pow_base[i] = (pow_base[i - 1] * base) % mod
        
        # For each node, we want to compute:
        #  f[node]: the hash of the DFS string starting from that node.
        #  r[node]: the hash of the reverse of the DFS string.
        #  size[node]: the length of the DFS string.
        #
        # The DFS procedure is defined as:
        #   For node x:
        #     for each child y (in increasing order) call dfs(y)
        #     append s[x] to the end.
        #
        # Thus, if node x has children c1, c2, ..., ck (in sorted order) then:
        #   S[x] = S[c1] + S[c2] + ... + S[ck] + s[x]
        #
        # And the reverse of S[x] is:
        #   reverse(S[x]) = s[x] + reverse(S[ck]) + ... + reverse(S[c1])
        
        f = [0] * n  # forward hash for S[node]
        r = [0] * n  # reverse hash for S[node]
        size = [0] * n  # length of S[node]
        answer = [False] * n
        
        def dfs(node: int) -> None:
            curr_forward = 0
            tot_len = 0
            # Process each child in increasing order.
            for child in children[node]:
                dfs(child)
                # When concatenating strings A (hash = curr_forward) and B (hash = f[child] and length = size[child]),
                # new_hash = (curr_forward * p^(len(B)) + f[child]) mod mod.
                curr_forward = (curr_forward * pow_base[size[child]] + f[child]) % mod
                tot_len += size[child]
            # Get the numeric value of the current node's character.
            letter_val = ord(s[node]) - ord('a') + 1
            # Append the current node's value to the end of the concatenated children string.
            f[node] = (curr_forward * pow_base[1] + letter_val) % mod
            size[node] = tot_len + 1

            # Now compute the reverse hash.
            # Since reverse(S[node]) = s[node] + reverse(S[child_k]) + ... + reverse(S[child_1])
            curr_reverse = letter_val
            for child in reversed(children[node]):
                curr_reverse = (curr_reverse * pow_base[size[child]] + r[child]) % mod
            r[node] = curr_reverse
            
            # The DFS string is a palindrome if its forward hash equals its reverse hash.
            answer[node] = (f[node] == r[node])
        
        # Start DFS from the root (node 0). This will compute the DFS string (and associated hash values)
        # for every node in the tree since the tree is connected.
        dfs(0)
        return answer


# --- Optional Testing ---
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    parent = [-1, 0, 0, 1, 1, 2]
    s = "aababa"
    # Expected: [True, True, False, True, True, True]
    print(sol.findAnswer(parent, s))
    
    # Example 2:
    parent = [-1, 0, 0, 0, 0]
    s = "aabcb"
    # Expected: [True, True, True, True, True]
    print(sol.findAnswer(parent, s))