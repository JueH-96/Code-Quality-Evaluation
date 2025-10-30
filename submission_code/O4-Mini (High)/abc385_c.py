import sys
def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    H = [int(next(it)) for _ in range(N)]
    # ans = maximum number of buildings in an AP with same height
    ans = 1
    # N1 = N-1, reuse many times
    N1 = N - 1
    # Try every possible spacing d
    # Once (N1//d + 1) <= ans, for larger d it will only decrease, so break
    for d in range(1, N):
        # Maximum possible length for this d (starting at index 0)
        max_len = N1 // d + 1
        if max_len <= ans:
            break
        # Try every starting position i
        # If from i the possible length <= ans, break inner loop
        # (it only gets worse as i increases)
        limit = N - d
        for i in range(limit):
            # quick check of remaining potential
            if (N1 - i) // d + 1 <= ans:
                break
            v = H[i]
            cnt = 1
            k = i + d
            # extend while same height
            # once mismatch, sequence must end
            while k < N and H[k] == v:
                cnt += 1
                k += d
            # update answer
            if cnt > ans:
                ans = cnt
                # if we've reached the max possible for this d,
                # no need to keep trying larger i for this d
                # since potential_i only decreases
                # but we'll hit the i-break above naturally
        # next d
    # print the result
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()