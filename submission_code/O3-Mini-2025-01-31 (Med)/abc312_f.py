# YOUR CODE HERE
def main():
    import sys, bisect
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    M = int(next(it))
    # Separate items into three groups:
    A = []  # pull-tab items, type0 (happiness)
    B = []  # regular canned items, type1 (happiness)
    C = []  # can opener items (capacity)
    for _ in range(n):
        t = int(next(it))
        x = int(next(it))
        if t == 0:
            A.append(x)
        elif t == 1:
            B.append(x)
        else:
            C.append(x)
    # sort descending
    A.sort(reverse=True)
    B.sort(reverse=True)
    C.sort(reverse=True)
    
    # Precompute prefix sums.
    def prefix_sum(arr):
        ps = [0]*(len(arr)+1)
        for i, val in enumerate(arr):
            ps[i+1] = ps[i] + val
        return ps
    prefixA = prefix_sum(A)  # pull-tabs
    prefixB = prefix_sum(B)  # regular cans
    prefixC = prefix_sum(C)  # openers (capacities)
    # Note: prefixC is increasing.
    
    best = 0
    # r is the number of regular cans selected.
    # r can range from 0 to min(M, len(B)).
    r_max = min(M, len(B))
    for r in range(r_max + 1):
        # For a fixed r, let a be the number of pull-tabs we will choose.
        # We have r+a food items; then number of openers = op = M - (r+a).
        # We want to maximize food happiness: prefixB[r] + prefixA[a].
        # Since prefixA is increasing, for fixed r, we want a as high as possible,
        # i.e. we want op as small as possible.
        # However, a cannot exceed len(A) and also we must have a <= M - r.
        # The maximum a is thus min(M - r, len(A)).
        # But note: op = M - r - a; if a = min(M - r, len(A)) then op is as small as possible.
        # However the openers we choose (op items from C) must satisfy:
        #    (i) op must be at least M - r - len(A) (so that a <= len(A))
        #    (ii) op <= M - r and op <= len(C) (cannot choose more openers than available)
        op_lower = max(0, M - r - len(A))
        op_upper = M - r
        if op_upper > len(C):
            op_upper = len(C)
        if op_lower > op_upper:
            continue
        # We need to choose an op in the range [op_lower, op_upper]
        # such that the total capacity of the chosen openers, prefixC[op],
        # is at least r (i.e. they can “open” r regular cans).
        # Among the allowed op values, we want the smallest op (to maximize a = M - r - op).
        pos = bisect.bisect_left(prefixC, r, op_lower, op_upper + 1)
        if pos == op_upper + 1:
            continue  # no valid opener choice for this r
        a = M - r - pos  # pull-tabs count chosen
        # a must be within [0, min(M - r, len(A))].
        if a < 0 or a > min(M - r, len(A)):
            continue
        candidate = prefixB[r] + prefixA[a]
        if candidate > best:
            best = candidate
    sys.stdout.write(str(best))
    
if __name__ == '__main__':
    main()