class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        from collections import deque
        INF = 10**15
        
        # 1) Combine duplicates by keeping only the minimum cost for each distinct word.
        cost_map = {}
        for w, c in zip(words, costs):
            if w not in cost_map:
                cost_map[w] = c
            else:
                cost_map[w] = min(cost_map[w], c)
        
        # 2) Build Aho-Corasick automaton for the distinct words in cost_map.
        #    Each node has: 
        #      - a next array of size 26 (for transitions),
        #      - a fail link,
        #      - an out list storing (length_of_word, cost).
        
        # We'll store transitions in a list of lists; -1 means no transition.
        trie_next = [[-1]*26]
        fail = [-1]
        out = [[]]  # out[node] will hold list of (word_length, cost)
        
        def add_word(word, cst):
            """Insert word into the trie, and add (len(word), cst) to the out list at terminal node."""
            node = 0
            for ch in word:
                idx = ord(ch) - ord('a')
                if trie_next[node][idx] == -1:
                    trie_next[node][idx] = len(trie_next)
                    trie_next.append([-1]*26)
                    fail.append(-1)
                    out.append([])
                node = trie_next[node][idx]
            out[node].append((len(word), cst))
        
        # Insert each distinct word into the trie
        for w, cst in cost_map.items():
            add_word(w, cst)
        
        # Build the fail links using BFS
        q = deque()
        # For root's children, fail link = 0 if they exist, else point them to root
        for ch in range(26):
            nxt = trie_next[0][ch]
            if nxt != -1:
                fail[nxt] = 0
                q.append(nxt)
            else:
                trie_next[0][ch] = 0  # fallback from root to root
        
        # BFS to construct fail links
        while q:
            u = q.popleft()
            f = fail[u]
            for ch in range(26):
                v = trie_next[u][ch]
                if v != -1:
                    fail[v] = trie_next[f][ch]
                    # Inherit "out" from the fail-linked node
                    out[v].extend(out[fail[v]])
                    q.append(v)
                else:
                    trie_next[u][ch] = trie_next[f][ch]
        
        # 3) Use DP to find minimum cost. dp[i] = min cost to form target[:i].
        n = len(target)
        dp = [INF]*(n+1)
        dp[0] = 0
        
        # Parse target with the automaton
        state = 0
        for i in range(n):
            c = ord(target[i]) - ord('a')
            # Move to next state
            state = trie_next[state][c]
            # For each pattern ending here, update dp
            for (length_word, cost_word) in out[state]:
                start_idx = i + 1 - length_word  # where this word started
                if start_idx >= 0:
                    dp[i+1] = min(dp[i+1], dp[start_idx] + cost_word)
        
        # 4) If dp[n] is INF, it's not possible; otherwise return dp[n]
        return dp[n] if dp[n] != INF else -1