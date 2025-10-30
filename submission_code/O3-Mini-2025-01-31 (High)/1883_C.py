def main():
    import sys

    data = sys.stdin.read().strip().split()
    if not data:
        return
    t = int(data[0])
    pos = 1
    out_lines = []

    # helper function: count the exponent of 2 in x
    def v2(x):
        cnt = 0
        while x % 2 == 0:
            cnt += 1
            x //= 2
        return cnt

    # Process each test case
    for _ in range(t):
        n = int(data[pos]); pos += 1
        k = int(data[pos]); pos += 1
        arr = list(map(int, data[pos:pos+n]))
        pos += n

        if k == 2:
            # For product to be divisible by 2, at least one number must be even.
            ans = 0
            for a in arr:
                if a % 2 == 0:
                    ans = 0
                    break
            else:
                ans = 1
            out_lines.append(str(ans))

        elif k == 3:
            # For product to be divisible by 3 (a prime), one number must be a multiple of 3.
            found = False
            best = 10**9
            for a in arr:
                if a % 3 == 0:
                    found = True
                    break
                else:
                    r = a % 3
                    # minimal increments needed to reach the next multiple of 3 is (3 - r)
                    cost = 3 - r
                    if cost < best:
                        best = cost
            out_lines.append("0" if found else str(best))

        elif k == 5:
            # For product to be divisible by 5, one number must be divisible by 5.
            found = False
            best = 10**9
            for a in arr:
                if a % 5 == 0:
                    found = True
                    break
                else:
                    r = a % 5
                    cost = 5 - r
                    if cost < best:
                        best = cost
            out_lines.append("0" if found else str(best))

        elif k == 4:
            # k == 4 means the product must have at least 2 factors of 2 (since 4 = 2^2).
            # Compute the baseline: sum of v2(a) over the array.
            base = 0
            for a in arr:
                base += v2(a)
            if base >= 2:
                out_lines.append("0")
            else:
                # We need extra factors: required = 2 - base (this is either 1 or 2)
                required = 2 - base
                INF = 10**9
                # For each element, we will see how many increments (operations) are needed to increase its contribution.
                # We want:
                #   best1: minimum x (>=1) such that f(a+x) - f(a) >= 1
                #   best2: minimum x (>=1) such that f(a+x) - f(a) >= 2
                best1_list = []
                best2_list = []
                for a in arr:
                    init = v2(a)
                    best1 = INF
                    best2 = INF
                    # Try a few increments (a is small; a<=10 so trying x from 1 to 15 is safe)
                    for x in range(1, 16):
                        new_val = a + x
                        new_exp = 0
                        temp = new_val
                        while temp % 2 == 0:
                            new_exp += 1
                            temp //= 2
                        delta = new_exp - init
                        if delta >= 1 and x < best1:
                            best1 = x
                        if delta >= 2 and x < best2:
                            best2 = x
                    best1_list.append(best1)
                    best2_list.append(best2)
                
                ans = INF
                if required == 1:
                    # Only one candidate modification that gives an extra factor (i.e. improvement >=1) is needed.
                    ans = min(best1_list)
                elif required == 2:
                    # There are two ways:
                    # 1. Pick a single element where the modification gives improvement>=2.
                    cand_single = min(best2_list)
                    ans = min(ans, cand_single)
                    # 2. Pick two elements, each giving at least improvement of 1.
                    valid = [cost for cost in best1_list if cost < INF]
                    if len(valid) >= 2:
                        valid.sort()
                        cand_pair = valid[0] + valid[1]
                        ans = min(ans, cand_pair)
                out_lines.append(str(ans))
        else:
            # According to the constraints, k is one of {2, 3, 4, 5}
            out_lines.append("0")
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()