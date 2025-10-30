from typing import List

class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        # Build children lists and sort them
        children = [[] for _ in range(n)]
        for i in range(1, n):
            p = parent[i]
            children[p].append(i)
        for ch in children:
            ch.sort()

        # We will compute for each node u:
        #   pidx[u] = the post-order index of u in the DFS of the whole tree
        #   l[u]    = the minimum post-order index among u and all nodes in u's subtree
        pidx = [0] * n
        l = [0] * n

        # Iterative post-order DFS
        stack = [(0, False)]
        post_index = 0
        while stack:
            u, visited = stack.pop()
            if not visited:
                # Push marker to process u after its children
                stack.append((u, True))
                # Push children in reverse so that smallest child is processed first
                for v in reversed(children[u]):
                    stack.append((v, False))
            else:
                # All children of u are done; assign post-order index
                r = post_index
                pidx[u] = r
                # Compute l[u] = min( r, l[child] for child in children[u] )
                min_l = r
                for v in children[u]:
                    if l[v] < min_l:
                        min_l = l[v]
                l[u] = min_l
                post_index += 1

        # Build the integer array of labels in post-order
        # v_arr[ idx ] = integer code of s[node] where idx = pidx[node]
        v_arr = [0] * n
        for node in range(n):
            # map 'a'..'z' to 1..26
            v_arr[pidx[node]] = ord(s[node]) - ord('a') + 1

        # Prepare reversed array for reverse-hash
        v_rev = v_arr[::-1]

        # Rolling hash parameters (double-mod for safety)
        mod1 = 10**9 + 7
        mod2 = 10**9 + 9
        base1 = 91138233
        base2 = 97266353

        # Precompute powers
        pow1_mod1 = [1] * (n + 1)
        pow2_mod2 = [1] * (n + 1)
        for i in range(n):
            pow1_mod1[i+1] = (pow1_mod1[i] * base1) % mod1
            pow2_mod2[i+1] = (pow2_mod2[i] * base2) % mod2

        # Prefix-hash for v_arr (forward)
        h1_fwd = [0] * (n + 1)
        h2_fwd = [0] * (n + 1)
        for i in range(n):
            h1_fwd[i+1] = (h1_fwd[i] * base1 + v_arr[i]) % mod1
            h2_fwd[i+1] = (h2_fwd[i] * base2 + v_arr[i]) % mod2

        # Prefix-hash for v_rev (forward on reversed array)
        h1_rev = [0] * (n + 1)
        h2_rev = [0] * (n + 1)
        for i in range(n):
            h1_rev[i+1] = (h1_rev[i] * base1 + v_rev[i]) % mod1
            h2_rev[i+1] = (h2_rev[i] * base2 + v_rev[i]) % mod2

        # Helper to get hash of v_arr[l..r] in O(1)
        def get_hash_fwd(h, pows, mod, l_pos, r_pos):
            length = r_pos - l_pos + 1
            res = h[r_pos+1] - h[l_pos] * pows[length] % mod
            if res < 0:
                res += mod
            return res

        # Now answer each query: substring v_arr[l[u]..pidx[u]] should be a palindrome
        ans = [False] * n
        for u in range(n):
            left = l[u]
            right = pidx[u]
            length = right - left + 1
            # forward hash on v_arr[left..right]
            hf1 = get_hash_fwd(h1_fwd, pow1_mod1, mod1, left, right)
            hf2 = get_hash_fwd(h2_fwd, pow2_mod2, mod2, left, right)
            # corresponding reversed segment in v_rev is [n-1-right .. n-1-left]
            rl = n - 1 - right
            rr = n - 1 - left
            hr1 = get_hash_fwd(h1_rev, pow1_mod1, mod1, rl, rr)
            hr2 = get_hash_fwd(h2_rev, pow2_mod2, mod2, rl, rr)
            if hf1 == hr1 and hf2 == hr2:
                ans[u] = True

        return ans