import sys
def main():
    import sys
    data = sys.stdin.readline
    line = data().split()
    if not line:
        return
    N = int(line[0]); X = int(line[1])
    A = list(map(int, data().split()))
    # Pair up values with original 1-based indices
    arr = [(A[i], i+1) for i in range(N)]
    arr.sort(key=lambda x: x[0])
    # Try fixing the first element of the triple
    for i in range(N-2):
        ai, idx_i = arr[i]
        # Prune by minimal and maximal possible sums
        # minimal sum with this ai
        mn = ai + arr[i+1][0] + arr[i+2][0]
        if mn > X:
            break
        # maximal sum with this ai
        mx = ai + arr[N-2][0] + arr[N-1][0]
        if mx < X:
            continue
        # Two‐pointer search for the other two
        target = X - ai
        l, r = i+1, N-1
        while l < r:
            s = arr[l][0] + arr[r][0]
            if s == target:
                idx_j = arr[l][1]
                idx_k = arr[r][1]
                i1, i2, i3 = idx_i, idx_j, idx_k
                # output in increasing order
                if i1 > i2: i1, i2 = i2, i1
                if i2 > i3: i2, i3 = i3, i2
                if i1 > i2: i1, i2 = i2, i1
                print(i1, i2, i3)
                return
            elif s < target:
                l += 1
            else:
                r -= 1
    # No triple found
    print(-1)

if __name__ == "__main__":
    main()