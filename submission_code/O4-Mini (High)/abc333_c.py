def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    # We need at least C(m+2,3) >= 333 sums; m=12 gives C(14,3)=364 >= 333
    m = 12
    # build repunits: rep[i] = integer with i ones, e.g., rep[3]=111
    rep = [0] * (m + 1)
    for i in range(1, m + 1):
        rep[i] = rep[i - 1] * 10 + 1

    sums = set()
    # generate all sums rep[i] + rep[j] + rep[k] with 1 <= i <= j <= k <= m
    for i in range(1, m + 1):
        for j in range(i, m + 1):
            for k in range(j, m + 1):
                sums.add(rep[i] + rep[j] + rep[k])

    sorted_sums = sorted(sums)
    # output the N-th smallest (1-based)
    print(sorted_sums[N - 1])

# call main
main()