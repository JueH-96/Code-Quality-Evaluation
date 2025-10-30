def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    # Read the toy sizes and box sizes.
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(n - 1)]

    # Sort both arrays. (Toys: A, Boxes: B)
    A.sort()
    B.sort()
    # We now have N toys with sizes A[0 ... n-1] (in non-decreasing order)
    # and n-1 boxes with sizes B[0 ... n-2].

    # The idea is to choose one toy (say A[i]) to be stored in the purchased box.
    # Then the remaining n-1 toys (which are A[0...i-1] and A[i+1...n-1])
    # must be assignable to the given boxes.
    # A necessary and sufficient condition is that, after sorting the remaining toys
    # in non-decreasing order, the k-th smallest toy (0-indexed) is <= B[k] for every k.
    #
    # When A is sorted, if we remove the element A[i], the remaining sorted list is:
    #   A[0], A[1], ..., A[i-1], A[i+1], ..., A[n-1]
    # We then need:
    #   for j in 0 .. i-1:  A[j] <= B[j],
    #   for j in i .. n-2:  A[j+1] <= B[j].
    #
    # We can precompute:
    #   pre[i]  = True if for all j in [0, i) we have A[j] <= B[j]. (Left part)
    #   suff[i] = True if for all k in [i, n-1) (i.e. for k=i,...,n-2)
    #             we have A[k+1] <= B[k]. (Right part)
    #
    # Then, candidate removal index i is valid if (pre[i] and suff[i]) holds.
    # In that case the purchased box must have size at least A[i].
    # Our answer is the smallest such A[i] (which is optimal since A is sorted).

    # Precompute prefix condition.
    pre = [True] * n  # pre[i] corresponds to checking A[0..i-1] with B[0..i-1].
    # For i = 0, no toy on the left so condition is True.
    for i in range(1, n):
        # Check that the (i-1)-th toy can go into box (i-1).
        # B is of length n-1 so index i-1 is valid for i in 1..n-1.
        pre[i] = pre[i - 1] and (A[i - 1] <= B[i - 1])
    
    # Precompute suffix condition.
    suff = [True] * n  # suff[i] will check that for all k from i to n-2, A[k+1] <= B[k].
    # For candidate removal i = n-1 (i.e. removing the largest toy), the right part is empty.
    suff[n - 1] = True
    for i in range(n - 2, -1, -1):
        suff[i] = (A[i + 1] <= B[i]) and suff[i + 1]

    # Now check for each candidate removal index i (0 <= i < n),
    # if both conditions hold, using the given boxes for the remaining toys.
    ans = -1
    for i in range(n):
        if pre[i] and suff[i]:
            ans = A[i]
            break

    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()