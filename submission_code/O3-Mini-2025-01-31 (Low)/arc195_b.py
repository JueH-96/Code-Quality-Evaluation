def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(n)]
    
    # The operations allowed:
    # 1. For any A[i] or B[i] that is –1, you can assign any nonnegative integer.
    # 2. You can rearrange the elements of A arbitrarily.
    # 
    # Our goal is to have a constant S such that for every position i in 1..N,
    #   A[i] + B[i] = S,
    # and all entries in A and B are nonnegative.
    #
    # Notice that reordering A gives us freedom to pair any fixed value of A with any
    # B (either fixed or to be determined). Thus, the only “hard” restrictions come
    # from the positions where both A[i] and B[i] are already fixed (i.e. not –1). 
    #
    # If any two positions have both fixed numbers, they force a value for S.
    # If these forced sums conflict, the answer is "No".
    #
    # Once we have S (or if none is forced, we can choose one), we require that for every
    # fixed A value, S - a is nonnegative so that we could set the corresponding B (if missing)
    # to that value; and similarly, for every fixed B value, S - b must be nonnegative.
    #
    # Algorithm:
    # 1. Scan positions where A[i] != –1 and B[i] != –1. Let candidate S be the sum.
    #    If more than one candidate exists and they differ, output "No".
    # 2. If no candidate is forced, choose some S large enough.
    #    A safe choice is S = (max fixed A value) + (max fixed B value). (If there are no fixed values, S can be 0.)
    # 3. For every fixed A, check that S - A[i] >= 0.
    #    Similarly, for every fixed B, check that S - B[i] >= 0.
    # 4. If all constraints are satisfied, then by filling in the missing entries as S - fixedValue
    #    and by reordering A as needed, the answer is "Yes".
    
    candidate = None
    for i in range(n):
        if A[i] != -1 and B[i] != -1:
            s = A[i] + B[i]
            if candidate is None:
                candidate = s
            elif candidate != s:
                sys.stdout.write("No")
                return

    if candidate is None:
        # Choose S as (max fixed A) + (max fixed B).
        maxA = max((a for a in A if a != -1), default=0)
        maxB = max((b for b in B if b != -1), default=0)
        candidate = maxA + maxB

    # Check that for every fixed A (or B), setting the paired value to candidate - value is nonnegative.
    for a in A:
        if a != -1 and candidate - a < 0:
            sys.stdout.write("No")
            return
    for b in B:
        if b != -1 and candidate - b < 0:
            sys.stdout.write("No")
            return
    
    sys.stdout.write("Yes")


if __name__ == '__main__':
    main()