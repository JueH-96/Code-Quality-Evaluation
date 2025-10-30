def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    t = int(data[0])
    pos = 1
    results = []
    
    # Explanation:
    # We are given an array A of length n and allowed operations that transfer one unit
    # from an element with larger index to one with a smaller index. In other words,
    # we can only move units “left‐wards”; note that index 1 (the first element) can only gain
    # and index n (the last element) can only lose.
    #
    # Thus if we want to achieve a non-decreasing final sequence B (B1 <= B2 <= ... <= Bn),
    # we must have:
    #   • B1 >= A1   (since the first element can only be increased)
    #   • Bn <= An   (since the last element can only be decreased)
    #
    # And because B is non-decreasing, every B[i] must lie in the range [B1, Bn] ⊆ [A1, An].
    # Consequently, the total sum of B – which equals the initial sum S = sum(A) because
    # the operations preserve the total – must satisfy:
    #
    #        n * A1 <= S <= n * An.
    #
    # It is not hard to see that this condition is not only necessary but also sufficient.
    # (In the “if” part one may observe that if S is in [n*A1, n*An] then there exists
    # some non-decreasing sequence B whose entries lie in [A1, An] and which sums to S.
    # Then by a simple flow argument one may show that B can be obtained from A by taking
    # units from some indices (which are on the right) and moving them to indices on the left.)
    #
    # One more remark:
    # If A1 > An then there is no chance to obtain B with B1 <= B2 <= ... <= Bn because
    # the only allowed moves increase A1 and decrease An.
    
    for _ in range(t):
        n = int(data[pos])
        pos += 1
        arr = list(map(int, data[pos:pos+n]))
        pos += n
        
        first = arr[0]
        last = arr[-1]
        total = sum(arr)
        
        # Check the necessary conditions:
        if first > last or total < n * first or total > n * last:
            results.append("No")
        else:
            results.append("Yes")
    
    sys.stdout.write("
".join(results))

if __name__ == '__main__':
    main()