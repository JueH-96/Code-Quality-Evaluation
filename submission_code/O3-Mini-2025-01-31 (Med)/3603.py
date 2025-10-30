from typing import List
import sys
sys.setrecursionlimit(300000)

class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        # Build the tree children list.
        children = [[] for _ in range(n)]
        for i in range(1, n):
            children[parent[i]].append(i)
        # Make sure the children are processed in increasing order.
        for i in range(n):
            children[i].sort()

        # We use rolling hash to “compress” the entire DFS result of a subtree.
        # Recall: The DFS order is defined as processing all children (in sorted order)
        # and then appending the current node's character. In other words,
        # for node u with children [c1, c2, ..., ck]:
        #
        #    F[u] = F[c1] + F[c2] + ... + F[ck] + s[u]
        #
        # Its reverse is:
        #    R[u] = s[u] + R[c_k] + ... + R[c2] + R[c1]
        #
        # The string is a palindrome exactly when F[u] equals R[u].
        # To combine hashes we need to know the lengths. Hence, for each subtree rooted at u
        # we compute a tuple: (f1, f2, L, r1, r2) where
        #    f1, f2: hash (pair of mod values) of F[u]
        #    L: length of F[u] (and R[u], obviously)
        #    r1, r2: hash of R[u]
        #
        # We choose two mod primes and a base.
        mod1 = 10**9+7
        mod2 = 10**9+9
        base = 137  # can be any prime base

        # Precompute powers up to maximum length n (since a subtree cannot have more than n nodes).
        maxLen = n + 1  # a safe bound
        pow1 = [1] * (maxLen + 1)
        pow2 = [1] * (maxLen + 1)
        for i in range(1, maxLen + 1):
            pow1[i] = (pow1[i-1] * base) % mod1
            pow2[i] = (pow2[i-1] * base) % mod2

        # answer[i] will be True if the DFS traversal string of subtree rooted at i is a palindrome.
        answer = [False] * n

        def dfs(u):
            # We'll accumulate child's DFS-results in order to “build” F[u].
            # Start with an empty hash (for F) and a total length of 0.
            f1, f2 = 0, 0
            L_total = 0
            child_vals = []  # store each child's computed tuple in order.
            for c in children[u]:
                val = dfs(c)
                child_vals.append(val)
                # Combine F[c] into our hash:
                child_len = val[2]
                f1 = (f1 * pow1[child_len] + val[0]) % mod1
                f2 = (f2 * pow2[child_len] + val[1]) % mod2
                L_total += child_len

            # Append the current node's character at the end.
            # Using mapping: a -> 1, b -> 2, ...
            cur = ord(s[u]) - ord('a') + 1
            f1 = (f1 * base + cur) % mod1
            f2 = (f2 * base + cur) % mod2
            L_total += 1

            # For building R[u], note that:
            # R[u] = s[u] + (R[child_k] + ... + R[child_1])
            r1, r2 = cur, cur
            # Go over children in reverse order.
            for val in reversed(child_vals):
                child_len = val[2]
                r1 = (r1 * pow1[child_len] + val[3]) % mod1
                r2 = (r2 * pow2[child_len] + val[4]) % mod2

            # If F[u] equals R[u] then the DFS string is a palindrome.
            if f1 == r1 and f2 == r2:
                answer[u] = True
            else:
                answer[u] = False

            return (f1, f2, L_total, r1, r2)

        # We compute DFS starting from the root (node 0). Since our recursion
        # computes the DFS result for every subtree, answer[] gets filled.
        dfs(0)
        return answer


# The code below is for local testing and reading input from standard input.
# It also serves as an example of how one may run the solution.
# Input is expected in the following format:
#    First token is the parent list as an array string, e.g.: "[-1,0,0,1,1,2]"
#    Second token is the string s.
#
# For example:
#    Input:
#       [-1,0,0,1,1,2]
#       aababa
#    Output:
#       [true,true,false,true,true,true]
#
def solve():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    # parse the parent list; remove square brackets and split by comma
    parent_str = data[0].strip()
    if parent_str[0] == "[" and parent_str[-1] == "]":
        parent_str = parent_str[1:-1]
    parent = list(map(int, parent_str.split(','))) if parent_str.strip() else []
    s = data[1] if len(data) > 1 else ""
    sol = Solution()
    ans = sol.findAnswer(parent, s)
    # Print result as lower-case booleans in a list form.
    out = []
    for val in ans:
        out.append("true" if val else "false")
    sys.stdout.write("[" + ",".join(out) + "]")

if __name__ == '__main__':
    solve()