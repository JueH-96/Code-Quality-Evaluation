def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    P = [int(next(it)) for _ in range(N)]

    # Pair each score with its original index, sort descending by score
    paired = [(P[i], i) for i in range(N)]
    paired.sort(key=lambda x: x[0], reverse=True)

    ans = [0] * N
    r = 1
    i = 0
    # Assign ranks in groups of equal scores
    while i < N:
        score = paired[i][0]
        j = i
        # find the group of equal scores
        while j < N and paired[j][0] == score:
            j += 1
        # assign current rank r to all in this group
        for k in range(i, j):
            _, idx = paired[k]
            ans[idx] = r
        # increment r by the size of the group
        r += (j - i)
        i = j

    # output the ranks in original order
    out = '
'.join(str(rank) for rank in ans)
    sys.stdout.write(out)

if __name__ == "__main__":
    main()