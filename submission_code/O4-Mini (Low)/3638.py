class Solution:
    def makeStringGood(self, s: str) -> int:
        # Count initial frequencies
        from collections import Counter
        cnt = Counter(s)
        # Precompute for each source letter i and target j the minimal cost per character:
        # - If i <= j: we can increment i to j in (j-i) steps
        # - Else: best is delete + insert = 2
        cost = [[0]*26 for _ in range(26)]
        for i in range(26):
            for j in range(26):
                if i <= j:
                    cost[i][j] = j - i
                else:
                    cost[i][j] = 2
        
        INF = 10**18
        n = len(s)
        best = INF
        
        # For each possible number of distinct letters k = 1..26
        for k in range(1, 27):
            # The target frequency per letter must make total length = k*f close to n
            # f ranges so that k*f is in [0, n + ops]; but ops >= |n - kf|
            # We only need f up to n (more than that only increases inserts).
            # So we try f from 0 to n//k + 1
            maxf = n // k + 1
            for f in range(maxf + 1):
                total_len = k * f
                # Cost to adjust total length
                len_cost = abs(n - total_len)
                # Now, among 26 letters we choose the best k letters to serve as targets.
                # For each candidate target letter j, compute the cost to gather f characters as:
                #   take cheapest f units from all sources i (up to cnt[i] copies), each at cost[i][j],
                #   and if not enough, use insert at cost=1.
                per_target = []
                for j in range(26):
                    # Gather all remappings from sources
                    bucket = []
                    for i in range(26):
                        if cnt[i] > 0:
                            bucket.append((cost[i][j], cnt[i]))
                    # We can also insert arbitrarily many at cost=1
                    # Now pick cheapest f units
                    bucket.sort()
                    need = f
                    csum = 0
                    for cst, avail in bucket:
                        use = min(avail, need)
                        csum += use * cst
                        need -= use
                        if need == 0:
                            break
                    # any leftover needs insertion
                    if need > 0:
                        csum += need * 1
                    per_target.append(csum)
                # pick k targets of minimal per_target cost
                per_target.sort()
                assign_cost = sum(per_target[:k])
                best = min(best, len_cost + assign_cost)
        return best