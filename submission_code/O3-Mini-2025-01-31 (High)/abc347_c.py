def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    A = int(next(it))
    B = int(next(it))
    T = A + B
    # Read the plan days and compute each day modulo T.
    D_list = [int(next(it)) for _ in range(N)]
    mods = [d % T for d in D_list]
    mods.sort()
    
    # The key observation is to note that if today is unknown then choosing a starting day x is
    # equivalent to choosing a cyclic shift. In our re‐formulation the condition is that there exists
    # some c ∈ [0,T) so that every point from mods is in the set {c, c+1, …, c+A-1} (all arithmetic mod T).
    # Equivalently, the N points (mods) on a circle of T integers must be coverable by an arc of length A.
    # A standard trick is to double the sorted array (adding T to each element in the second copy)
    # so that a wrap‐around interval becomes a contiguous one in the doubled list.
    
    doubled = mods + [x + T for x in mods]
    
    # We now look for an index i (0 ≤ i < N) such that the interval [doubled[i], doubled[i] + A - 1]
    # contains at least N points.
    # Because our holiday set is exactly A consecutive days (from c to c+A-1), if this is possible then
    # there is a way to choose today's day so that all (x + D_i) mod T come out in the holiday block.
    
    j = 0
    for i in range(N):
        # Holiday block if we start at doubled[i] is [doubled[i], doubled[i] + A - 1]
        end_point = doubled[i] + A - 1
        # Move pointer j so that all points from i until j-1 are within the block.
        # We only need to consider i + N points in the doubled list.
        while j < i + N and j < len(doubled) and doubled[j] <= end_point:
            j += 1
        if j - i >= N:
            sys.stdout.write("Yes")
            return
    sys.stdout.write("No")

if __name__ == '__main__':
    main()