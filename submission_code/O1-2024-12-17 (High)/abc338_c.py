def main():
    import sys
    import math

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    Q = list(map(int, data[1:1+N]))
    A = list(map(int, data[1+N:1+2*N]))
    B = list(map(int, data[1+2*N:1+3*N]))

    # Compute the maximum number of dish A we could possibly make if we only made A
    # (ignoring ingredients where A_i = 0, since they do not limit dish A)
    maxA = float('inf')
    for i in range(N):
        if A[i] > 0:
            maxA = min(maxA, Q[i] // A[i])
    # Similarly, compute the maximum number of dish B we could possibly make if we only made B
    maxB = float('inf')
    for i in range(N):
        if B[i] > 0:
            maxB = min(maxB, Q[i] // B[i])

    # If either is still inf (which should not happen if the input constraints are followed),
    # set them to 0 just in case, but by problem statement there's always at least one Ai>0
    # and one Bi>0, so these won't remain inf.
    if maxA == float('inf'):
        maxA = 0
    if maxB == float('inf'):
        maxB = 0

    # Upper bound on total dishes is at most making all A plus all B
    # (This is safe because we can never exceed that sum in any combination.)
    high = maxA + maxB
    low = 0
    ans = 0

    def feasible(total):
        # We want to check if there exists an integer x in [0..total]
        # such that for all i,  A[i]*x + B[i]*(total - x) <= Q[i].
        # Rewrite: B[i]*total + (A[i] - B[i]) * x <= Q[i].
        # Let M = Q[i] - B[i]*total, and d = A[i] - B[i].
        # Then d*x <= M.  If d>0 => x <= M/d, if d<0 => x >= M/d, if d=0 => B[i]*total <= Q[i].
        lo_x = 0
        hi_x = total
        for i in range(N):
            a = A[i]
            b = B[i]
            q = Q[i]
            M = q - b*total
            d = a - b
            if d == 0:
                # Then we need b*total <= q
                if b*total > q:
                    return False
                # No further restriction on x from this ingredient
            elif d > 0:
                # x <= floor(M/d)
                # If M/d is negative and fairly large in magnitude, we might quickly get hi_x < lo_x
                val = math.floor(M / d)
                if val < lo_x:
                    return False
                hi_x = min(hi_x, val)
                if lo_x > hi_x:
                    return False
            else:
                # d < 0 => x >= ceil(M/d)
                val = math.ceil(M / d)
                if val > hi_x:
                    return False
                lo_x = max(lo_x, val)
                if lo_x > hi_x:
                    return False

        # Finally, x must be in [0..total] as well
        if lo_x > total or hi_x < 0:
            return False
        lo_x = max(lo_x, 0)
        hi_x = min(hi_x, total)
        return lo_x <= hi_x

    # Binary search for the maximum total in [0..high]
    while low <= high:
        mid = (low + high) // 2
        if feasible(mid):
            ans = mid
            low = mid + 1
        else:
            high = mid - 1

    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()