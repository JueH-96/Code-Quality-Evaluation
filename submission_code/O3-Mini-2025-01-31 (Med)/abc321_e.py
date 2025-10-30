def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    t = int(data[0])
    out_lines = []
    # Precompute powers of 2 for fast shifting (enough for our d values up to about 70)
    pow2 = [1 << i for i in range(70)]
    
    # Given a node "a" (root of a subtree in the infinite complete binary tree)
    # and an integer d, the set of nodes at exactly d levels below a (0-indexed: d=0 means node a itself)
    # in an unconstrained infinite binary tree is all integers in the interval:
    #    [a * 2^d, a * 2^d + 2^d - 1]
    # Here we want to count only those nodes that are <= N.
    def count_all(a, d, N):
        if a > N:
            return 0
        if d == 0:
            return 1  # just the node a
        L = a * pow2[d]
        if L > N:
            return 0
        R = a * pow2[d] + (pow2[d] - 1)
        if R > N:
            R = N
        return R - L + 1

    # Explanation of our approach:
    # For each vertex Y whose distance from X is exactly K, the unique simple path between X and Y
    # can be thought of as: first, take u upward moves (from 0 up to the depth of X)
    # and then take v downward moves (with u + v = K). Let A be the vertex reached after u upward moves.
    # Then, from A we choose a downward path of length v. For u > 0, there is a restriction: we cannot go
    # down the branch that leads back to X because that would retrace our upward moves.
    #
    # The downward moves from a node A, unconstrained, result in nodes with indices in the interval:
    #   [A * 2^v, A * 2^v + 2^v - 1]   (if they exist in our tree up to N)
    # Hence, we use count_all(A, v, N) to count those nodes.
    # And when there is a forbidden branch (when u > 0) we must subtract the count of nodes that lie
    # in the downward subtree of the child of A that lies on the path from A to X.
    # If u > 0, then the forbidden child is Pchild = X >> (u-1)
    #
    # Finally, we sum for all u = 0 to min(K, depth(X)). (The depth of X is X.bit_length()-1.)

    idx = 1
    for _ in range(t):
        N = int(data[idx]); idx += 1
        X = int(data[idx]); idx += 1
        K = int(data[idx]); idx += 1

        # For K==0, only X itself qualifies.
        if K == 0:
            out_lines.append("1")
            continue

        # depth of X (root has depth 0)
        dX = X.bit_length() - 1  # because numbers [2^L, 2^(L+1)-1] have depth L
        total_ans = 0
        # We can only go upward at most dX times (else we hit an invalid node).
        max_u = dX if dX < K else K
        for u in range(max_u + 1):
            A = X >> u          # The vertex reached after u upward moves.
            v = K - u           # Then we must go down v steps.
            if v == 0:
                total_ans += 1  # exactly vertex A
            else:
                if u == 0:
                    # Starting from X itself without any upward move:
                    total_ans += count_all(A, v, N)
                else:
                    # For u>0, we must avoid going back the way we came.
                    # The forbidden branch from A is Pchild = X >> (u - 1)
                    Pchild = X >> (u - 1)
                    total_ans += (count_all(A, v, N) - count_all(Pchild, v - 1, N))
        out_lines.append(str(total_ans))
    sys.stdout.write("
".join(out_lines))


if __name__ == '__main__':
    main()