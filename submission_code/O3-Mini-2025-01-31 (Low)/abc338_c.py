def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data: 
        return
    it = iter(input_data)
    N = int(next(it))
    Q = [int(next(it)) for _ in range(N)]
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N)]
    
    # Given s total servings = x servings of dish A and (s - x) servings of dish B,
    # the constraint for ingredient i is:
    #   A[i]*x + B[i]*(s - x) <= Q[i]
    # Rearranging:
    #   (A[i] - B[i])*x <= Q[i] - B[i]*s
    #
    # Note: x must be an integer in [0, s]. Depending on the sign of (A[i]-B[i]),
    # we get lower or upper constraints on x as follows:
    #
    # 1) If A[i] == B[i]: Then the condition becomes A[i]*s <= Q[i].
    #    (If not satisfied, no x can fix that.)
    #
    # 2) If A[i] < B[i]: Then (A[i]-B[i]) is negative.
    #    Multiply by -1 to reverse inequality:
    #         (B[i]-A[i])*x >= B[i]*s - Q[i].
    #    Hence, we need:
    #         x >= ceil((B[i]*s - Q[i]) / (B[i]-A[i])),
    #    but if B[i]*s <= Q[i], then there is no lower constraint from this ingredient.
    #
    # 3) If A[i] > B[i]: Then (A[i]-B[i]) positive so:
    #         x <= floor((Q[i]-B[i]*s) / (A[i]-B[i])).
    
    # For a candidate total serving count s, we need to see if there is some
    # integer x in [0, s] that satisfies all ingredient constraints.
    def feasible(s):
        low, high_bound = 0, s  # x must lie in [0, s]
        for i in range(N):
            a = A[i]
            b = B[i]
            qi = Q[i]
            if a == b:
                # Condition becomes: a * s <= qi
                if a * s > qi:
                    return False
            elif a < b:
                # Calculate the lower bound on x:
                diff = b - a  # positive
                # We require:
                #    x >= ceil((b*s - qi) / diff) if b*s > qi, otherwise no restriction.
                if b * s > qi:
                    # Ceiling division: (num + diff - 1) // diff
                    req = (b * s - qi + diff - 1) // diff
                    low = max(low, req)
                # else, no constraint from this ingredient
            else: # a > b:
                # Calculate the upper bound on x:
                diff = a - b  # positive
                # We require:
                #    x <= floor((qi - b*s) / diff)
                possible = (qi - b * s) // diff
                high_bound = min(high_bound, possible)
            
            # If the range for x is empty, then s is not feasible.
            if low > high_bound:
                return False
        return True
    
    # We want the maximum total number s such that there exists some valid x.
    # A safe upper bound is sum(Q), since even in the best case every serving uses at least 1 gram.
    lo = 0
    hi = sum(Q) + 1  # hi is exclusive
    
    # Binary search for the maximum s for which feasible(s) is true.
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            lo = mid + 1
        else:
            hi = mid
    
    # Maximum s is lo - 1
    print(lo - 1)

if __name__ == '__main__':
    main()