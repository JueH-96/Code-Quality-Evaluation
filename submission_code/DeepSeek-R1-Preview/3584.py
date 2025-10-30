import bisect
from collections import defaultdict

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        next_char = defaultdict(list)
        for idx, c in enumerate(word1):
            next_char[c].append(idx)
        
        n = len(word2)
        if n == 0:
            return []
        
        INF = float('inf')
        dp = [[INF] * 2 for _ in range(n + 1)]
        dp[0][0] = -1  # Starting state
        
        parent = [[None for _ in range(2)] for _ in range(n + 1)]
        
        for i in range(n):
            for s in [0, 1]:
                if dp[i][s] == INF:
                    continue
                current_j = dp[i][s]
                target = word2[i]
                
                # Option A: match without substitution
                if target in next_char:
                    lst = next_char[target]
                    pos = bisect.bisect_right(lst, current_j)
                    if pos < len(lst):
                        j = lst[pos]
                        if j < dp[i+1][s]:
                            dp[i+1][s] = j
                            parent[i+1][s] = (i, s, False)
                
                # Option B: use substitution if allowed
                if s == 0:
                    j = current_j + 1
                    if j < len(word1):
                        if j < dp[i+1][1]:
                            dp[i+1][1] = j
                            parent[i+1][1] = (i, s, True)
        
        # Determine the final state
        possible = []
        if dp[n][0] != INF:
            possible.append( (dp[n][0], 0) )
        if dp[n][1] != INF:
            possible.append( (dp[n][1], 1) )
        
        if not possible:
            return []
        
        # Choose the state with the smallest index
        possible.sort()
        final_index, final_s = possible[0]
        
        # Backtrack to find the sequence
        sequence = []
        current_i = n
        current_s = final_s
        
        while current_i > 0:
            if parent[current_i][current_s] is None:
                return []
            i_prev, s_prev, substitution = parent[current_i][current_s]
            sequence.append(dp[current_i][current_s])
            current_i = i_prev
            current_s = s_prev
        
        # The sequence is built in reverse order
        sequence.reverse()
        
        # Check if the formed string is almost equal
        # Since the DP ensures at most one substitution, this check is redundant but safe
        s = ''.join(word1[i] for i in sequence)
        mismatches = sum(1 for a, b in zip(s, word2) if a != b)
        if mismatches > 1:
            return []
        
        return sequence