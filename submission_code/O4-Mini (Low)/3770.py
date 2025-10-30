class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        N = n + m - 1
        
        # Initialize the word with '?' and apply 'T' constraints
        word = ['?'] * N
        for i, ch in enumerate(str1):
            if ch == 'T':
                # force word[i:i+m] == str2
                for j in range(m):
                    p = i + j
                    if word[p] == '?' or word[p] == str2[j]:
                        word[p] = str2[j]
                    else:
                        return ""  # conflict
        
        # Prepare the list of F-window indices covering each position
        # and initialize counters for each F-window
        # We'll index windows by their start i (0 <= i < n where str1[i]=='F')
        fwins = []    # list of (i) for each F-window
        win_idx = {}  # map from i to index in fwins
        for i, ch in enumerate(str1):
            if ch == 'F':
                win_idx[i] = len(fwins)
                fwins.append(i)
        W = len(fwins)
        
        # For each F-window, maintain questionCount and matchingCount
        qcnt = [m] * W
        mcnt = [0] * W
        
        # Build pos2wins: for each position, which F-windows include it
        pos2wins = [[] for _ in range(N)]
        for w_i, start in enumerate(fwins):
            for j in range(m):
                p = start + j
                pos2wins[p].append(w_i)
        
        # Initialize counts using the forced letters from 'T' windows
        for w_i, start in enumerate(fwins):
            qc = m
            mc = 0
            for j in range(m):
                p = start + j
                c = word[p]
                if c == '?':
                    continue
                qc -= 1
                if c == str2[j]:
                    mc += 1
            qcnt[w_i] = qc
            mcnt[w_i] = mc
        
        # Now fill the '?' positions greedily
        for p in range(N):
            if word[p] != '?':
                continue
            placed = False
            # try letters 'a' to 'z'
            for c in map(chr, range(97, 123)):
                # check if placing c at p would fully match any F-window
                violate = False
                for w_i in pos2wins[p]:
                    start = fwins[w_i]
                    j = p - start
                    new_q = qcnt[w_i] - 1
                    new_m = mcnt[w_i] + (1 if c == str2[j] else 0)
                    # violation if no more '?' and full match
                    if new_q == 0 and new_m == m:
                        violate = True
                        break
                if violate:
                    continue
                # commit c
                word[p] = c
                for w_i in pos2wins[p]:
                    start = fwins[w_i]
                    j = p - start
                    qcnt[w_i] -= 1
                    if c == str2[j]:
                        mcnt[w_i] += 1
                placed = True
                break
            if not placed:
                return ""
        
        # After filling, all T are satisfied and no F-window fully matches
        return "".join(word)