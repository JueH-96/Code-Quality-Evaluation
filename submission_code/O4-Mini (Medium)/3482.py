from typing import List
from collections import deque

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        # Deduplicate words, keep minimal cost
        word_cost = {}
        for w, c in zip(words, costs):
            if w in word_cost:
                if c < word_cost[w]:
                    word_cost[w] = c
            else:
                word_cost[w] = c

        # Build Aho-Corasick automaton
        # Each node: dict children, int fail, int out_link, list output (one tuple (length, cost))
        children = []
        fail = []
        out_link = []
        output = []

        def new_node():
            children.append({})
            fail.append(0)
            out_link.append(-1)
            output.append([])
            return len(children) - 1

        root = new_node()
        # Insert words
        for w, c in word_cost.items():
            node = root
            for ch in w:
                if ch not in children[node]:
                    nxt = new_node()
                    children[node][ch] = nxt
                node = children[node][ch]
            # store the word's length and cost at this output node
            output[node].append((len(w), c))

        # Build fail links with BFS
        q = deque()
        # Initialize fail of depth-1 nodes to root
        for ch, nxt in children[root].items():
            fail[nxt] = root
            # out_link: if fail node has output, link to fail, else inherit fail's out_link
            if output[fail[nxt]]:
                out_link[nxt] = fail[nxt]
            else:
                out_link[nxt] = out_link[fail[nxt]]
            q.append(nxt)

        while q:
            u = q.popleft()
            for ch, v in children[u].items():
                # Compute fail for v
                f = fail[u]
                while f != root and ch not in children[f]:
                    f = fail[f]
                if ch in children[f] and children[f][ch] != v:
                    f = children[f][ch]
                fail[v] = f
                # Compute out_link for v
                if output[f]:
                    out_link[v] = f
                else:
                    out_link[v] = out_link[f]
                q.append(v)

        # DP array
        n = len(target)
        INF = 10**18
        dp = [INF] * (n + 1)
        dp[0] = 0

        # Traverse the target string
        state = root
        for j, ch in enumerate(target):
            # Aho-Corasick transition
            while state != root and ch not in children[state]:
                state = fail[state]
            if ch in children[state]:
                state = children[state][ch]
            else:
                state = root

            # Check matches at this state and follow out_links
            temp = state
            while temp != -1:
                if output[temp]:
                    # For each word ending here
                    for length, cost in output[temp]:
                        start = j - length + 1
                        if start >= 0 and dp[start] != INF:
                            dp[j+1] = min(dp[j+1], dp[start] + cost)
                temp = out_link[temp]

        return dp[n] if dp[n] != INF else -1