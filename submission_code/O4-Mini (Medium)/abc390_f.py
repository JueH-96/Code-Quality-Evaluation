import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    A = [0] * (N + 1)
    for i in range(1, N + 1):
        A[i] = int(next(it))

    # SumD: sum over subarrays of number of distinct values
    last = [0] * (N + 1)
    sumD = 0
    for i in range(1, N + 1):
        x = A[i]
        prev = last[x]
        # number of subarrays where this A[i] is first occurrence of x
        cnt = (i - prev) * (N - i + 1)
        sumD += cnt
        last[x] = i

    # Build positions lists
    pos = [[] for _ in range(N + 2)]
    for i in range(1, N + 1):
        pos[A[i]].append(i)

    total_sub = N * (N + 1) // 2

    def missing(positions):
        # number of subarrays that miss all values in `positions`
        if not positions:
            return total_sub
        res = 0
        prev = 0
        for p in positions:
            gap = p - prev - 1
            if gap > 0:
                res += gap * (gap + 1) // 2
            prev = p
        # after last
        gap = N - prev
        if gap > 0:
            res += gap * (gap + 1) // 2
        return res

    # SumC: sum over v of #subarrays containing both v and v+1
    sumC = 0
    # we will merge pos[v] and pos[v+1] on the fly
    for v in range(1, N):
        p1 = pos[v]
        p2 = pos[v+1]
        # missing individually
        miss1 = missing(p1)
        miss2 = missing(p2)
        # missing both: merge p1 and p2
        # classic two-pointer merge
        merged = []
        i = j = 0
        len1 = len(p1)
        len2 = len(p2)
        while i < len1 and j < len2:
            if p1[i] < p2[j]:
                merged.append(p1[i]); i += 1
            else:
                merged.append(p2[j]); j += 1
        if i < len1:
            merged.extend(p1[i:])
        if j < len2:
            merged.extend(p2[j:])
        miss_both = missing(merged)
        # subarrays containing both = total - miss1 - miss2 + miss_both
        sumC += (total_sub - miss1 - miss2 + miss_both)

    # answer
    ans = sumD - sumC
    print(ans)

if __name__ == "__main__":
    main()