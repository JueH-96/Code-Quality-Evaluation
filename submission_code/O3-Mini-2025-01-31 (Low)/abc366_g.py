def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    
    # Build the degree list.
    degree = [0] * n
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        # Convert to 0-index.
        degree[u - 1] += 1
        degree[v - 1] += 1

    # The condition is:
    # For every vertex v (with at least one edge), the XOR of the numbers on its neighbors must be 0.
    # Consider the simplest assignment where a connected component is assigned a constant value "a".
    # Then, for each vertex v, the XOR of its neighbors equals:
    #    a XOR a XOR ... (deg(v) times).
    # If deg(v) is even then a XOR a XOR ... XOR a (even number of copies) equals 0.
    # However, if deg(v) is odd then the XOR equals a (which is nonzero because we must choose a from [1,2^60-1])
    # and the condition fails.
    #
    # Consequently, for any vertex with degree>=1 we require that its degree is even.
    #
    # If any vertex with an edge (non-isolated) has an odd degree then there is no possible assignment.
    for d in degree:
        if d % 2 == 1:
            sys.stdout.write("No")
            return

    # Otherwise, one valid solution is to assign every vertex a nonzero number.
    # For every vertex that is not isolated (degree>=1), a uniform assignment works because,
    # with an even number of neighbors, the XOR equals 0.
    # For isolated vertices we can assign any number since they have no condition.
    # We'll use the number 1 (which lies in [1,2^60 - 1]) for all vertices.
    assignment = ["1"] * n
    sys.stdout.write("Yes
" + " ".join(assignment))


if __name__ == '__main__':
    main()