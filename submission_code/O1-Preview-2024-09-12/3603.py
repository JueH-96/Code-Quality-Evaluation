class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        import sys
        sys.setrecursionlimit(1 << 25)
        n = len(s)
        tree = [[] for _ in range(n)]
        for i in range(1, n):
            p = parent[i]
            tree[p].append(i)

        P1 = 911
        mod1 = 10 ** 9 + 7
        P2 = 3571
        mod2 = 10 ** 9 + 9

        maxlen = n + 5
        powP1 = [1] * maxlen
        powP2 = [1] * maxlen
        for i in range(1, maxlen):
            powP1[i] = (powP1[i - 1] * P1) % mod1
            powP2[i] = (powP2[i - 1] * P2) % mod2

        len_arr = [0] * n
        hash1 = [0] * n
        rev_hash1 = [0] * n
        hash2 = [0] * n
        rev_hash2 = [0] * n

        answer = [False] * n

        def dfs(x):
            len_x = 0
            h1 = 0
            h2 = 0
            rev_h1 = ord(s[x]) % mod1
            rev_h2 = ord(s[x]) % mod2
            len_x += 1  # for s[x]
            children = tree[x]
            # Hash for forward direction (post-order traversal)
            # Process children in increasing order
            for y in children:
                dfs(y)
                # h1 = (h1 * P1^{len[y]} + hash1[y]) % mod1
                h1 = (h1 * powP1[len_arr[y]] + hash1[y]) % mod1
                h2 = (h2 * powP2[len_arr[y]] + hash2[y]) % mod2
                len_x += len_arr[y]
            # Append s[x]
            h1 = (h1 * P1 + ord(s[x])) % mod1
            h2 = (h2 * P2 + ord(s[x])) % mod2

            # Reverse hash
            # Process children in decreasing order
            for y in reversed(children):
                # rev_h1 = (rev_h1 * P1^{len[y]} + rev_hash1[y]) % mod1
                rev_h1 = (rev_h1 * powP1[len_arr[y]] + rev_hash1[y]) % mod1
                rev_h2 = (rev_h2 * powP2[len_arr[y]] + rev_hash2[y]) % mod2
            len_arr[x] = len_x
            hash1[x] = h1
            hash2[x] = h2
            rev_hash1[x] = rev_h1
            rev_hash2[x] = rev_h2

        # Precompute all hashes and lengths
        dfs_all_called = [False] * n

        def compute_all_hashes(x):
            dfs(x)
            dfs_all_called[x] = True
            answer[x] = (hash1[x] == rev_hash1[x] and hash2[x] == rev_hash2[x])
            # Process children
            for y in tree[x]:
                if not dfs_all_called[y]:
                    compute_all_hashes(y)

        # Since the tree is connected, we can start from any node
        for i in range(n):
            if not dfs_all_called[i]:
                compute_all_hashes(i)

        return answer