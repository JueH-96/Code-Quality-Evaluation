class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        L = n + m - 1                        # length of the wanted word
        
        # 1. place all compulsory characters coming from the 'T' positions
        word = ['?'] * L
        for i, tf in enumerate(str1):
            if tf == 'T':
                for j, ch in enumerate(str2):
                    p = i + j                # absolute position in word
                    if word[p] == '?':
                        word[p] = ch
                    elif word[p] != ch:      # contradictory information
                        return ""
        
        # 2. prepare data structures for the 'F' constraints
        #    satisfied[i]   – whether F-constraint starting at i is already satisfied
        #    free_left[i]   – how many still unassigned (‘?’) positions remain in this substring
        satisfied  = [True] * n
        free_left  = [0]    * n
        involved   = [[] for _ in range(L)]  # for every position p -> list of F–indices using p (only if not yet satisfied)
        
        for i, tf in enumerate(str1):
            if tf == 'F':
                mismatch_found = False
                free_positions = []
                for j in range(m):
                    p = i + j
                    if word[p] == '?':
                        free_positions.append(p)
                    elif word[p] != str2[j]:
                        mismatch_found = True
                if mismatch_found:
                    satisfied[i] = True
                else:
                    satisfied[i]  = False
                    free_left[i]  = len(free_positions)
                    if free_left[i] == 0:        # all fixed and equal -> impossible
                        return ""
                    for p in free_positions:     # register the open positions
                        involved[p].append(i)
        
        # 3. fill the remaining '?' greedily from left to right
        for p in range(L):
            if word[p] != '?':
                continue                        # already fixed by a 'T'
            
            placed = False
            for k in range(26):                 # try 'a' .. 'z'
                c = chr(ord('a') + k)
                ok = True
                for fi in involved[p]:
                    if satisfied[fi]:
                        continue
                    rel = p - fi                # j position inside str2 (0..m-1)
                    if c == str2[rel]:
                        if free_left[fi] == 1:  # would become the last position and still equal -> invalid
                            ok = False
                            break
                if ok:
                    # commit the character and update the bookkeeping
                    word[p] = c
                    for fi in involved[p]:
                        if satisfied[fi]:
                            continue
                        rel = p - fi
                        if c == str2[rel]:
                            free_left[fi] -= 1
                        else:                   # mismatch created, constraint satisfied
                            satisfied[fi] = True
                    placed = True
                    break
            if not placed:                      # no letter works here
                return ""
        
        # 4. final verification (safety – should already hold)
        for i, tf in enumerate(str1):
            if tf == 'F' and not satisfied[i]:
                return ""
        
        return ''.join(word)