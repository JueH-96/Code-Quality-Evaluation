def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    N = int(data[0])
    # For each flavor, store the top two deliciousness values.
    # best[f] = [max1, max2] where max1 is the highest S for flavor f,
    # and max2 is the second highest S for flavor f.
    best = {}
    idx = 1
    for _ in range(N):
        f = int(data[idx])
        s = int(data[idx+1])
        idx += 2
        if f not in best:
            best[f] = [s, 0]
        else:
            if s > best[f][0]:
                best[f][1] = best[f][0]
                best[f][0] = s
            elif s > best[f][1]:
                best[f][1] = s

    # Candidate 1: Choose two cups with distinct flavors.
    # For each flavor we only consider the best cup, and then we choose the top two values.
    max1 = -1
    max2 = -1
    for f, pair in best.items():
        if pair[0] > max1:
            max2 = max1
            max1 = pair[0]
        elif pair[0] > max2:
            max2 = pair[0]
    res_diff = -1
    if len(best) >= 2:
        res_diff = max1 + max2

    # Candidate 2: Choose two cups from the same flavor.
    # Satisfaction = s + (t/2) where s is the maximum and t is the second maximum.
    res_same = -1
    for f, (first, second) in best.items():
        if second > 0:  # There are at least two cups for this flavor.
            candidate = first + (second // 2)
            if candidate > res_same:
                res_same = candidate

    # The maximum achievable satisfaction is the maximum of these two candidates.
    ans = res_diff if res_diff > res_same else res_same
    print(ans)

if __name__ == '__main__':
    main()