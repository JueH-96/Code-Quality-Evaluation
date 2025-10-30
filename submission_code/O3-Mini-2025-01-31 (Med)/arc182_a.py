def main():
    import sys,sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    mod = 998244353
    it = iter(data)
    try:
        N = int(next(it))
        Q = int(next(it))
    except StopIteration:
        return
    # Read Q operations: each op is (P, V)
    ops = []
    for _ in range(Q):
        try:
            p = int(next(it))
            v = int(next(it))
        except StopIteration:
            break
        ops.append((p, v))
    
    res = 0
    # k goes from 0 to N.
    # For a fixed boundary k, we force that every op with p <= k is done as a left–update
    # and every op with p > k is a right–update. These two “sub–sequences” (taken in the original order)
    # must have non–decreasing V–values.
    #
    # Note: k==0 means left–ops are empty (and that’s fine) and all ops are right–updates.
    # Similarly, k==N means all ops are left–updates.
    for k in range(0, N+1):
        valid_left = True
        last = 0  # initially S[1]=0
        for (p, v) in ops:
            if p <= k:  # this op (if chosen as left update) will update S[1]
                if v < last:
                    valid_left = False
                    break
                last = v
        if not valid_left:
            continue
        valid_right = True
        last = 0  # initially S[N]=0
        for (p, v) in ops:
            if p > k:  # op will be done as right update (updates S[N])
                if v < last:
                    valid_right = False
                    break
                last = v
        if valid_right:
            res += 1
    res %= mod
    sys.stdout.write(str(res))
    
if __name__ == '__main__':
    main()