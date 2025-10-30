def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    t = int(data[0])
    pos = 1

    # Given k is one of {2,3,4,5}:
    # If k = 2, product divisible by 2 means at least one factor 2.
    # k = 3  => at least one factor 3.
    # k = 4  => at least two factors 2.
    # k = 5  => at least one factor 5.
    # So for each test, we only care about one prime factor (p) with a required exponent.
    def get_prime_and_requirement(k):
        if k == 2:
            return 2, 1
        elif k == 3:
            return 3, 1
        elif k == 4:
            return 2, 2
        elif k == 5:
            return 5, 1

    # helper: count exponent of prime p in x
    def exp_in(x, p):
        cnt = 0
        while x % p == 0:
            cnt += 1
            x //= p
        return cnt

    # For each element a, we can add x (x operations) to get a new value (a+x).
    # The gain for prime p is: gain = exponent(a+x, p) - exponent(a, p), if positive.
    # We want to choose some elements to update (each element can be updated at most once with a chosen x)
    # to increase the overall exponent of p in the product by enough amount (if needed) with minimal total cost.
    #
    # Since n can be large but k (and so the required exponent) is small, we will do the following:
    #   For each element, try x from 0 to some maximum (say 50; given a up to 10, 50 is safe).
    #   Record, for each improvement (clipped to at most req) what is the minimal cost (number of increments) needed.
    #
    # Then we run a small knapSack-like DP (deficit will be at most 2) over these candidates.
    
    res_lines = []
    for _ in range(t):
        n = int(data[pos]); pos += 1
        k = int(data[pos]); pos += 1
        arr = list(map(int, data[pos: pos+n]))
        pos += n

        p, req = get_prime_and_requirement(k)
        
        # First, count the total exponent of p in the product without any operations.
        curr_total = 0
        base_exps = []
        for a in arr:
            e = exp_in(a, p)
            base_exps.append(e)
            curr_total += e
        
        # If already divisible, no operations needed.
        deficit = max(0, req - curr_total)
        if deficit == 0:
            res_lines.append("0")
            continue

        # For each element, try to improve its p-exponent by adding x (from 0 to 50)
        # and record the best (minimal cost) for each possible improvement (only needed up to req).
        candidates = []  # will be a list of dicts: { improvement_value : cost }
        for a, base in zip(arr, base_exps):
            best_for_imp = {}  # improvement -> minimal cost
            # We try small increments: x from 0 to 50
            for x in range(0, 51):
                new_val = a + x
                new_exp = exp_in(new_val, p)
                improvement = new_exp - base
                # Only consider positive improvements.
                if improvement <= 0:
                    continue
                # We only need to consider improvement values up to req, as extra doesn't add extra benefit.
                improvement = improvement if improvement <= req else req
                if improvement not in best_for_imp or x < best_for_imp[improvement]:
                    best_for_imp[improvement] = x
            if best_for_imp:
                candidates.append(best_for_imp)

        # Now, we have candidates (each candidate represents an update option on one element).
        # We wish to choose a subset of these updates to obtain at least "deficit" total improvement at minimal total cost.
        # Since deficit is very small (max 2) we can use a simple DP.
        INF = 10**9
        dp = [0] + [INF] * req  # dp[i] = minimal cost to achieve i improvement
        for cand in candidates:
            new_dp = dp[:]  # make a copy for this candidate
            for imp, cost in cand.items():
                # Try to update dp in reverse order to not reuse a candidate more than once.
                for have in range(req, -1, -1):
                    new_imp = have + imp
                    if new_imp > req:
                        new_imp = req
                    if dp[have] + cost < new_dp[new_imp]:
                        new_dp[new_imp] = dp[have] + cost
            dp = new_dp
        answer = dp[req]
        res_lines.append(str(answer))
    sys.stdout.write("
".join(res_lines))


if __name__ == '__main__':
    main()