class Solution:
    def XORHelp(self, A, i, j, results, lcp):
        res = 0
        if i < 0 or j < 0:
            return 0
        while True:
            if i == j:
                res ^= A[i]
                break
            if lcp[i][j] == 0:
                res ^= A[i]
                res ^= A[j]
                break
            nw = lcp[i][j] - 1
            res ^= results[nw][i]
            res ^= results[nw][j + 1]
            i -= 1 << nw
            j -= 1 << nw
        return res

    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        queries = [x[::-1] for x in queries]
        m = len(nums)
        n = len(queries)
        down = [nums]
        lcp = [[0] * m for _ in range(17)]
        last = nums
        for i in range(17):
            this = [0] * (m - (1 << i))
            for j in range(m - (1 << i)):
                this[j] = last[j] ^ last[j + (1 << i)]
            lcp[i] = [0] * (m - (1 << i))
            for j in range(m - 1 - (1 << i)):
                if this[j] == this[j + 1]:
                    lcp[i][j] = lcp[i - 1][j] + 1
            if this:
                last = this[:]
            down.append(last[:])
        # lcp
        results = down[1:]
        # tr
        tr = [sorted(list(range(len(down[i]))), key=lambda x: down[i][x]) for i in range(1, 18)]
        # bin
        for i in range(n):
            l, r = queries[i]
            queries[i][1] = bisect_left(tr[1], r-l) + (r-l)
            queries[i][0] = bisect_left(tr[1], l)
        queries = sorted(queries)
        max = [0] * 17
        ans = []
        for i in range(n):
            while queries[i][1] >= n and n > 0:
                first = down[1][tr[1][n - 1]]
                last = tr[1][n - 1]
                n -= 1
                for j in range(1, 17):
                    x = down[j][first]
                    if tr[j].index(first) == n:
                        continue
                    first = tr[j][n]
                    t = tr[j].index(first)
                    L = t - 1
                    while L >= 0 and down[j][tr[j][L]] > x:
                        L -= 1
                    R = t + 1
                    while R < len(tr[j]) and down[j][tr[j][R]] > x: 
                        R += 1
                    if R == len(tr[j]):
                        tr[j][t] = last
                    else:
                        tr[j][t:R] = [last] + tr[j][t:R]
                    if L > -1 and down[j][tr[j][L]] == x:
                        tr[j][L: t] = tr[j][L: t] + [tr[j][L]]
            while queries[i][0] < 0 and 0 < len(tr[0]):
                for j in range(17):
                    t = tr[j][0]
                    if tr[j][0] + (1 << j) >= len(down[j]):
                        del tr[j][0]
                        continue
                    tr[j][0] = tr[j][0] + (1 << j)
                    tr[j][t] = t + (1 << j)
            if n == 0 and len(tr[0]) == 0:
                ans.append(0)
                continue
            # tr
            val = 0
            for j in range(16, -1, -1):
                if len(tr[j]) == 0:
                    continue
                if n > 0 and tr[j][n - 1] + val >= len(down[j]):
                    continue
                k = bisect_left(tr[j], (n - 1) & ((1 << j) - 1))
                x = down[j][tr[j][k] + val]
                # print(A[i], j, n - 1, k, tr[j], val, x)
                if x != 0:
                    val += (1 << j)
                n = k
            ans.append(val)
        return [self.XORHelp(nums, 0, q[1], results, lcp) ^ self.XORHelp(nums, 0, q[0] - 1, results, lcp) ^ x for q, x in zip(queries, ans)]