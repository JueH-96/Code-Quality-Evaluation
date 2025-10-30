from typing import List
from collections import deque

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        # Build Aho-Corasick automaton for the given words.
        # Each trie node will have:
        #   edges: dict(char -> next_state)
        #   fail: the fallback (failure) link
        #   output: list of (word_len, cost) for each pattern ending at this node
        class AhoNode:
            def __init__(self):
                self.edges = {}
                self.fail = 0
                self.output = []  # list of (word_len, cost)
        
        # Build trie
        trie = [AhoNode()]
        for w, c in zip(words, costs):
            current = 0
            for ch in w:
                if ch not in trie[current].edges:
                    trie[current].edges[ch] = len(trie)
                    trie.append(AhoNode())
                current = trie[current].edges[ch]
            # At this node, we record the pattern that ends here
            trie[current].output.append((len(w), c))
        
        # Build failure links using BFS
        q = deque()
        # For single-character transitions from root:
        for ch, nxt in trie[0].edges.items():
            trie[nxt].fail = 0
            q.append(nxt)
        
        # BFS to build fail links
        while q:
            state = q.popleft()
            fail_state = trie[state].fail
            # For each edge from 'state', follow fail links
            for ch, nxt in trie[state].edges.items():
                q.append(nxt)
                f = fail_state
                # Follow fail links if we do not have the transition
                while f > 0 and ch not in trie[f].edges:
                    f = trie[f].fail
                if ch in trie[f].edges:
                    f = trie[f].edges[ch]
                trie[nxt].fail = f
                # Also merge output lists (patterns) from the failed-to state
                trie[nxt].output += trie[f].output
        
        n = len(target)
        INF = float('inf')
        dp = [INF] * (n + 1)
        dp[0] = 0
        
        # Traverse the target with the automaton
        current_state = 0
        for i, ch in enumerate(target):
            # Move to next state in automaton
            while current_state > 0 and ch not in trie[current_state].edges:
                current_state = trie[current_state].fail
            if ch in trie[current_state].edges:
                current_state = trie[current_state].edges[ch]
            else:
                current_state = 0
            
            # For each pattern ending here, update dp
            for length_w, cost_w in trie[current_state].output:
                start_index = i + 1 - length_w
                if start_index >= 0 and dp[start_index] != INF:
                    dp[i + 1] = min(dp[i + 1], dp[start_index] + cost_w)
        
        return dp[n] if dp[n] != INF else -1