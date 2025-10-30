import sys, math

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    Q = [int(next(it)) for _ in range(N)]
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N)]
    
    # Given we want to produce x servings of dish A and y servings of dish B
    # with total s = x + y, each ingredient i must satisfy:
    #   x*A[i] + y*B[i] <= Q[i].
    # Writing y = s - x, this becomes:
    #   s*B[i] + x*(A[i] - B[i]) <= Q[i]
    #
    # For a fixed total s, we want to know if some integer x in [0, s] exists
    # that simultaneously satisfies for every ingredient i:
    #   s*B[i] + x*(A[i] - B[i]) <= Q[i].
    #
    # Let d = A[i] - B[i]. Then:
    #   if d > 0:  x <= (Q[i] - s*B[i]) // d    (and Q[i]-s*B[i] must be nonnegative)
    #   if d < 0:  x >= ceil((Q[i] - s*B[i]) / d)
    #   if d == 0: then we must simply have s*B[i] <= Q[i] (or else no choice of x works).
    #
    # We then take the intersection of the constraints for x (always needing 0 <= x <= s).
    # If the intersection is nonempty then s servings can be produced.
    
    def feasible(s):
        lower_bound = 0  # x must be >= 0
        upper_bound = s  # and x must be <= s
        for i in range(N):
            d = A[i] - B[i]
            # Check the case when d > 0: constraint gives an upper bound on x.
            if d > 0:
                if Q[i] - s * B[i] < 0:
                    return False
                ub = (Q[i] - s * B[i]) // d
                upper_bound = min(upper_bound, ub)
            # When d < 0: constraint gives a lower bound.
            elif d < 0:
                lb = math.ceil((Q[i] - s * B[i]) / d)
                lower_bound = max(lower_bound, lb)
            # When d == 0, the inequality reduces to s * B[i] <= Q[i].
            else:
                if s * B[i] > Q[i]:
                    return False
        return lower_bound <= upper_bound

    # Binary search for the maximum total number of servings s.
    # An upper bound can be chosen safely as sum(Q), since every serving uses some ingredient.
    lo = 0
    hi = sum(Q) + 1  # hi is an exclusive upper bound
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            lo = mid + 1
        else:
            hi = mid
    ans = lo - 1
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()