def main():
    import sys
    from bisect import bisect_left, bisect_right

    data = sys.stdin
    line = data.readline().split()
    if not line:
        return
    N, Q = map(int, line)
    A = list(map(int, data.readline().split()))
    
    queries = []
    for i in range(Q):
        r, x = map(int, data.readline().split())
        queries.append((r, x, i))
    # Sort queries by prefix R
    queries.sort(key=lambda t: t[0])
    
    tails = []      # tails[k] = minimal tail value of an increasing subseq of length k+1
    res = [0]*Q
    curR = 0
    
    for r, x, qi in queries:
        # Extend the prefix to r, updating the LIS tails
        while curR < r:
            v = A[curR]
            pos = bisect_left(tails, v)
            if pos == len(tails):
                tails.append(v)
            else:
                tails[pos] = v
            curR += 1
        # Answer is the largest length l with tails[l-1] <= x
        ans = bisect_right(tails, x)
        res[qi] = ans
    
    out = sys.stdout
    w = out.write
    for ans in res:
        w(str(ans))
        w('
')

if __name__ == "__main__":
    main()