def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    X = int(next(it))
    U = [0] * n
    D = [0] * n
    S = [0] * n
    total = 0
    min_sum = 10**18  # a large number
    
    for i in range(n):
        u = int(next(it))
        d = int(next(it))
        U[i] = u
        D[i] = d
        s = u + d
        S[i] = s
        total += s
        if s < min_sum:
            min_sum = s

    # For a candidate H, we must have for every tooth i: 
    #   u_i (final upper tooth length) in [0, U[i]] and final lower tooth length is H - u_i in [0, D[i]].
    # So u_i must lie in the interval:
    #   [max(0, H - D[i]), min(U[i], H)]
    # Also, the chain condition demands that for consecutive teeth, |u_i - u_(i+1)| <= X.
    # We now check if there exists a sequence u_1,...,u_n with u_i in the available interval for tooth i and
    # satisfying |u_i - u_(i+1)| <= X.
    #
    # The key observation is that if we can choose these u_i, then the cost for this candidate H is:
    #   cost = sum(S[i] - H for i in range(n)) = total - n * H.
    # Since cost decreases with H, we wish to maximize H.
    # However, H must be at most S[i] for every i and we can only lower the length.
    # Thus, H must be in the range [0, min(S[i])]. We will binary search on H.
    
    def feasible(H):
        # We'll compute the running "reachable" interval of possible u values from left to right.
        # For tooth i, the allowed u_i is:
        #     [l_i, r_i] = [max(0, H - D[i]), min(U[i], H)]
        # Additionally, if the previous tooth had a possible value in [L, R],
        # then the next tooth must be in [L - X, R + X] (to satisfy |u_i - u_(i+1)| <= X).
        low = max(0, H - D[0])
        high = min(U[0], H)
        if low > high:
            return False
        for i in range(1, n):
            cur_low = max(0, H - D[i])
            cur_high = min(U[i], H)
            if cur_low > cur_high:
                return False
            # Because u_i from previous tooth can vary between low and high,
            # u_(i) should also lie in the range [low - X, high + X].
            allowed_low = low - X
            allowed_high = high + X
            # The new interval for this tooth is the intersection of its intrinsic interval and the allowed range.
            new_low = max(cur_low, allowed_low)
            new_high = min(cur_high, allowed_high)
            if new_low > new_high:
                return False
            low, high = new_low, new_high
        return True

    loH = 0
    hiH = min_sum
    best = -1
    while loH <= hiH:
        mid = (loH + hiH) // 2
        if feasible(mid):
            best = mid
            loH = mid + 1  # try a higher H since cost is lower when H is higher
        else:
            hiH = mid - 1

    if best < 0:
        best = 0  # theoretically H = 0 is always feasible

    # Total cost = sum(S[i]) - n * best
    ans = total - n * best
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()