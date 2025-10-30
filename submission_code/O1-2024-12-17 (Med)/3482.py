class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        from collections import deque
        
        # Build Aho-Corasick Trie
        # -------------------------------------------------
        # Each trie node will have:
        #   trie[node][c] = next state for character c
        #   fail[node] = failure link
        #   output[node] = list of (word_length, cost) for words ending at this node
        
        # 1. Calculate the total number of characters across all words for trie size
        total_chars = sum(len(w) for w in words)
        max_states = total_chars + 1  # +1 for root
        
        # Initialize trie structures
        # Using lists for speed. -1 indicates no edge
        # We'll map 'a'..'z' -> 0..25
        trie = [[-1] * 26 for _ in range(max_states)]
        fail = [0] * max_states
        output = [[] for _ in range(max_states)]
        
        # Helper to convert character to index
        def char_to_index(c):
            return ord(c) - ord('a')
        
        # Build the trie with the given words
        state_count = 1  # next available state index, 0 is root
        for w_i, w in enumerate(words):
            cost_w = costs[w_i]
            curr = 0  # start at root
            for c in w:
                idx = char_to_index(c)
                if trie[curr][idx] == -1:
                    trie[curr][idx] = state_count
                    state_count += 1
                curr = trie[curr][idx]
            # Add (length of word, cost) to output of this ending node
            output[curr].append((len(w), cost_w))
        
        # Build the failure links using BFS
        q = deque()
        # For each letter from root, if it doesn't exist, link to root
        for c in range(26):
            nxt = trie[0][c]
            if nxt != -1:
                fail[nxt] = 0
                q.append(nxt)
            else:
                trie[0][c] = 0  # loop back to root if missing
        
        # BFS to build fail links & merge output lists
        while q:
            st = q.popleft()
            f = fail[st]
            for c in range(26):
                nxt = trie[st][c]
                if nxt != -1:
                    fail[nxt] = trie[f][c]
                    # Merge the output from fail-link node
                    output[nxt].extend(output[fail[nxt]])
                    q.append(nxt)
                else:
                    trie[st][c] = trie[f][c]

        # Now run DP over the target string using the built automaton
        n = len(target)
        INF = 10**15
        dp = [INF] * (n + 1)
        dp[0] = 0
        
        curr_state = 0
        for i in range(n):
            c_idx = char_to_index(target[i])
            # Transition in Aho-Corasick automaton
            curr_state = trie[curr_state][c_idx]
            
            # For each pattern that ends here, update dp
            for (length_word, cost_word) in output[curr_state]:
                start_idx = i + 1 - length_word
                if start_idx >= 0:
                    if dp[start_idx] != INF:
                        new_cost = dp[start_idx] + cost_word
                        if new_cost < dp[i + 1]:
                            dp[i + 1] = new_cost
        
        return dp[n] if dp[n] != INF else -1