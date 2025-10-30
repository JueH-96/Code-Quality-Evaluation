def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    A = list(map(int, input_data[2:2+N]))
    B = list(map(int, input_data[2+N:2+2*N]))

    # If M=2, the only values are 0 and 1, and a "good" sequence of length >=2
    # must alternate. In that scenario, if A != B, we cannot make any single change
    # without causing two adjacent elements to collide. Hence it's only possible
    # if A == B already; otherwise impossible.
    if M == 2:
        if A == B:
            print(0)
        else:
            print(-1)
        return

    # Otherwise, for M >= 3, it turns out we can always achieve B from A by carefully
    # detouring around neighbors to avoid collisions. The minimal number of operations
    # is simply the sum of the minimal cyclic distances between corresponding elements.
    #
    # The distance between x and y mod M is:
    #   dist(x,y) = min( (x-y) mod M, (y-x) mod M )
    # which is the same as min(|x - y|, M - |x - y|) in normal arithmetic.
    #
    # We compute this sum and output it.

    def mod_dist(x, y, m):
        diff = abs(x - y)
        return min(diff, m - diff)

    total_ops = 0
    for i in range(N):
        total_ops += mod_dist(A[i], B[i], M)

    print(total_ops)

# Do not forget to call main()
main()