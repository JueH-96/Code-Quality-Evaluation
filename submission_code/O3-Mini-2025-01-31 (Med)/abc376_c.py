def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(n - 1)]
    
    # Sort the toy sizes and box sizes.
    A.sort()
    B.sort()
    m = len(B)  # = n-1

    # Given a candidate x (size of the purchased box), we now have a total of n boxes,
    # namely the m boxes in B and the extra box of size x.
    # We need to decide if we can assign each toy (sorted increasingly in A)
    # to a distinct box (also considered in non-decreasing order) so that
    # each toy's size is at most the capacity of the box it is placed in.
    #
    # The following function simulates a greedy matching algorithm.
    # We “merge” B and the extra box (of size x) in sorted order without forming
    # an extra list. The idea is to iterate through a total of n boxes in sorted order.
    # We have two “sources”:
    #   - The extra box: a single element with value x.
    #   - The m already available boxes: stored in B sorted.
    # In a merging process, if the extra box is not yet used and either
    # (a) we have exhausted B or (b) x is no larger than the next available B[j],
    # we choose the extra box.
    # For each box taken in that order we try to assign it to the smallest toy (in A)
    # that has not been assigned yet. (If a box’s capacity is smaller than the next toy’s requirement,
    # we skip that box.) If we are able to assign all n toys, the candidate x works.
    def can_assign(x):
        i = 0  # pointer for toys in A (n elements)
        j = 0  # pointer for boxes in B (m elements)
        used_extra = False  # whether we have used the purchased box
        
        # There are exactly n boxes to process (m from B plus one extra box).
        # We simulate the sorted order merge without creating an extra list.
        for _ in range(n):
            # Decide which box appears next in sorted order.
            if not used_extra and (j >= m or x <= B[j]):
                # Use the extra box here.
                box_cap = x
                used_extra = True
            else:
                # Use the next box from B.
                box_cap = B[j]
                j += 1
            # Greedy matching:
            # If the current box can accommodate the current toy, “assign” the toy.
            if i < n and box_cap >= A[i]:
                i += 1
                if i == n:  # All toys have been assigned.
                    break
        return (i == n)

    # We now seek the smallest integer x (with x > 0) such that we can assign all toys.
    # Note that if x is too small then even after merging the extra box,
    # some toy wouldn’t have a sufficiently large box.
    # On the other hand, increasing x can only help.
    #
    # What is a good upper bound for x? Every toy must go into some box.
    # In any valid assignment, the toy assigned to the purchased box requires that x is at least as large as that toy.
    # So certainly, if we try x = max(max(A), max(B)), it is the “largest” candidate we need to consider.
    # (If B is empty, use max(A).)
    if m > 0:
        high_bound = max(max(A), B[-1])
    else:
        high_bound = max(A)
    
    lo = 1
    hi = high_bound
    ans = -1
    found = False
    # Standard binary search for the minimal x.
    while lo <= hi:
        mid = (lo + hi) // 2
        if can_assign(mid):
            ans = mid
            found = True
            hi = mid - 1
        else:
            lo = mid + 1

    sys.stdout.write(str(ans) if found else "-1")

if __name__ == '__main__':
    main()