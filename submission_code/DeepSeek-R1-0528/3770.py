class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        L = n + m - 1
        
        ts = [i for i, ch in enumerate(str1) if ch == 'T']
        if ts:
            Z = [0] * m
            if m > 1:
                l, r = 0, 0
                for i in range(1, m):
                    if i <= r:
                        Z[i] = min(r - i + 1, Z[i - l])
                    else:
                        Z[i] = 0
                    while i + Z[i] < m and str2[Z[i]] == str2[i + Z[i]]:
                        Z[i] += 1
                    if i + Z[i] - 1 > r:
                        l = i
                        r = i + Z[i] - 1
            n_ts = len(ts)
            for i in range(n_ts):
                for j in range(i+1, n_ts):
                    d = ts[j] - ts[i]
                    if d >= m:
                        break
                    if Z[d] < m - d:
                        return ""
        
        ans = [None] * L
        for i in range(n):
            if str1[i] == 'T':
                for k in range(m):
                    pos = i + k
                    if ans[pos] is not None:
                        if ans[pos] != str2[k]:
                            return ""
                    else:
                        ans[pos] = str2[k]
        
        active_conditions = set()
        rem_count = {}
        aff_cond = [[] for _ in range(L)]
        
        for j in range(n):
            if str1[j] == 'F':
                satisfied = False
                free_positions = []
                for k in range(m):
                    pos = j + k
                    if ans[pos] is not None:
                        if ans[pos] != str2[k]:
                            satisfied = True
                    else:
                        free_positions.append(pos)
                if satisfied:
                    continue
                if not free_positions:
                    return ""
                active_conditions.add(j)
                rem_count[j] = len(free_positions)
                for pos in free_positions:
                    aff_cond[pos].append(j)
        
        for p in range(L):
            if ans[p] is not None:
                continue
            candidate_found = False
            for c in "abcdefghijklmnopqrstuvwxyz":
                valid = True
                for j in aff_cond[p]:
                    if j not in active_conditions:
                        continue
                    k = p - j
                    if c == str2[k] and rem_count[j] == 1:
                        valid = False
                        break
                if not valid:
                    continue
                ans[p] = c
                for j in aff_cond[p]:
                    if j not in active_conditions:
                        continue
                    k = p - j
                    if c == str2[k]:
                        rem_count[j] -= 1
                    else:
                        active_conditions.discard(j)
                candidate_found = True
                break
            if not candidate_found:
                return ""
        return "".join(ans)