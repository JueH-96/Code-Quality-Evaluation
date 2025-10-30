from collections import deque
from typing import List

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        # Build Aho-Corasick automaton for words, storing (length, cost) at output
        # Deduplicate words by taking minimal cost for identical words
        word_cost = {}
        for w, c in zip(words, costs):
            if w not in word_cost or c < word_cost[w]:
                word_cost[w] = c

        # Trie node structure: children dict, failure link, output list
        trie = []
        # each node: { 'ch': {}, 'fail': int, 'out': [(length, cost), ...] }
        def new_node():
            return {'ch': {}, 'fail': 0, 'out': []}
        trie.append(new_node())

        # Build trie
        for w, c in word_cost.items():
            node = 0
            for ch in w:
                if ch not in trie[node]['ch']:
                    trie[node]['ch'][ch] = len(trie)
                    trie.append(new_node())
                node = trie[node]['ch'][ch]
            # at word end, record its length and cost
            trie[node]['out'].append((len(w), c))

        # Build failure links
        q = deque()
        # Initialize depth-1 nodes: fail to 0
        for ch, nxt in trie[0]['ch'].items():
            trie[nxt]['fail'] = 0
            q.append(nxt)
        # BFS
        while q:
            v = q.popleft()
            for ch, nxt in trie[v]['ch'].items():
                # compute failure for nxt
                f = trie[v]['fail']
                while f and ch not in trie[f]['ch']:
                    f = trie[f]['fail']
                if ch in trie[f]['ch']:
                    f = trie[f]['ch'][ch]
                trie[nxt]['fail'] = f
                # merge output
                trie[nxt]['out'].extend(trie[f]['out'])
                q.append(nxt)

        n = len(target)
        INF = 10**18
        dp = [INF] * (n + 1)
        dp[0] = 0

        # Traverse target, update dp by matches ending at i
        node = 0
        for i, ch in enumerate(target):
            # follow trie edges, or failure links
            while node and ch not in trie[node]['ch']:
                node = trie[node]['fail']
            if ch in trie[node]['ch']:
                node = trie[node]['ch'][ch]
            # for all outputs at this node
            for length, cost in trie[node]['out']:
                start = i - length + 1
                if start >= 0 and dp[start] != INF:
                    dp[i+1] = min(dp[i+1], dp[start] + cost)
            # also carry forward the possibility of not ending a word here:
            # (no direct carry needed; dp[i+1] starts INF unless updated by a word)
        
        return dp[n] if dp[n] < INF else -1