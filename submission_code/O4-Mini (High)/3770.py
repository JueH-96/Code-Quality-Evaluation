class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        L = n + m - 1
        
        # 1. Start with all unknowns and apply 'T' constraints
        word = ['?'] * L
        for i, ch in enumerate(str1):
            if ch == 'T':
                # substring at i must equal str2
                for j in range(m):
                    p = i + j
                    c2 = str2[j]
                    if word[p] == '?' or word[p] == c2:
                        word[p] = c2
                    else:
                        # conflict
                        return ""
        
        # 2. Prepare data for 'F' intervals
        isF = [False] * n
        satisfied = [False] * n  # whether we already have a mismatch in that interval
        U = [0] * n              # count of unknowns remaining in that interval
        
        for i, ch in enumerate(str1):
            if ch == 'F':
                isF[i] = True
                sat = False
                cnt = 0
                # scan the substring word[i..i+m-1]
                for j in range(m):
                    p = i + j
                    if word[p] == '?':
                        cnt += 1
                    else:
                        # if it differs, we already satisfy the "not equal" condition
                        if word[p] != str2[j]:
                            sat = True
                satisfied[i] = sat
                U[i] = cnt
                # if it's unsatisfied and no unknowns to break equality, impossible
                if not sat and cnt == 0:
                    return ""
        
        # 3. forbidden counts: for each position p and each letter a-z,
        #    how many critical intervals force that letter to be forbidden there
        forbid_count = [[0] * 26 for _ in range(L)]
        
        # initialize critical intervals (U[i] == 1) before any assignment
        for i in range(n):
            if isF[i] and not satisfied[i] and U[i] == 1:
                # find the sole unknown in [i..i+m-1]
                for j in range(m):
                    p = i + j
                    if word[p] == '?':
                        req = str2[j]
                        forbid_count[p][ord(req) - 97] += 1
                        break
        
        # 4. Fill in the remaining '?' positions in lexicographic order
        for p in range(L):
            if word[p] == '?':
                # pick the smallest letter not forbidden here
                pick = None
                row = forbid_count[p]
                for ci in range(26):
                    if row[ci] == 0:
                        pick = chr(ci + 97)
                        break
                if pick is None:
                    return ""
                word[p] = pick
                
                # update all 'F' intervals covering position p
                lo = p - m + 1
                if lo < 0: lo = 0
                hi = p if p < n - 1 else n - 1
                for i in range(lo, hi + 1):
                    if not isF[i] or satisfied[i]:
                        continue
                    j = p - i
                    req = str2[j]
                    if pick == req:
                        # this is a match, so one unknown is resolved here
                        old_u = U[i]
                        U[i] = old_u - 1
                        # if we just dropped to U==1, it becomes critical
                        if U[i] == 1:
                            # find the only remaining unknown
                            for j2 in range(m):
                                p2 = i + j2
                                if word[p2] == '?':
                                    req2 = str2[j2]
                                    forbid_count[p2][ord(req2) - 97] += 1
                                    break
                        # we never let U go to 0 on an unsatisfied interval
                    else:
                        # mismatch => we satisfy this F-interval
                        # if it was critical (U==1), remove its pending forbid
                        if U[i] == 1:
                            # the forbidden letter at this p was req
                            forbid_count[p][ord(req) - 97] -= 1
                        satisfied[i] = True
        
        # 5. Final check (all F intervals must have at least one mismatch)
        for i in range(n):
            if isF[i] and not satisfied[i]:
                # if still unsatisfied, the final string must differ somewhere
                ok = False
                for j in range(m):
                    if word[i + j] != str2[j]:
                        ok = True
                        break
                if not ok:
                    return ""
        
        return "".join(word)