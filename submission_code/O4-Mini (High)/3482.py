from typing import List

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        # 1) Keep only the minimum cost for each distinct word
        min_cost = {}
        for w, c in zip(words, costs):
            if w not in min_cost or c < min_cost[w]:
                min_cost[w] = c

        # 2) Build a reversed‐trie of the words
        # children[node][ci] = next node index or -1
        # end_cost[node] = cost of the word ending at this node (None if none)
        children = [[-1] * 26]
        end_cost = [None]

        for w, c in min_cost.items():
            node = 0
            for ch in reversed(w):
                ci = ord(ch) - ord('a')
                nxt = children[node][ci]
                if nxt == -1:
                    nxt = len(children)
                    children[node][ci] = nxt
                    children.append([-1] * 26)
                    end_cost.append(None)
                node = nxt
            # record the cost at the terminal node
            if end_cost[node] is None or c < end_cost[node]:
                end_cost[node] = c

        # 3) DP over prefixes of target
        n = len(target)
        INF = 10**18
        dp = [INF] * (n + 1)
        dp[0] = 0

        # convert target to integer array for fast indexing
        arr = [ord(ch) - ord('a') for ch in target]
        children_local = children
        end_cost_local = end_cost

        for j in range(1, n + 1):
            best = INF
            node = 0
            # try to match any word ending at position j by scanning backwards
            for k in range(j - 1, -1, -1):
                node = children_local[node][arr[k]]
                if node == -1:
                    break
                cost_here = end_cost_local[node]
                if cost_here is not None:
                    v = dp[k] + cost_here
                    if v < best:
                        best = v
            dp[j] = best

        return dp[n] if dp[n] < INF else -1