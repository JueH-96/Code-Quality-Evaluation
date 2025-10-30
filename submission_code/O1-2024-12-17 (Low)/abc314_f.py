def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    p = [0]*(N-1)
    q = [0]*(N-1)
    idx = 1
    for i in range(N-1):
        p[i] = int(input_data[idx]); q[i] = int(input_data[idx+1])
        idx += 2

    MOD = 998244353

    # ------------------------------------------------------
    # Precompute modular inverses of [1..N] under MOD
    # We only need up to N because (a+b) <= N
    inv = [0]*(N+1)
    inv[1] = 1
    for i in range(2, N+1):
        inv[i] = (MOD - MOD//i)*inv[MOD%i] % MOD

    # ------------------------------------------------------
    # DSU with "val[x]" to store partial offsets.
    # If x is a root (parent[x] == x), val[x] is the "offset"
    #   that every node in x's component should add to get
    #   its final expected value.
    # If x is not a root, val[x] is the difference from x
    #   to its parent.
    parent = list(range(N+1))
    size = [1]*(N+1)
    val = [0]*(N+1)  # partial difference or component-offset

    def find(x):
        """Find with path-compression, updating val[x] to be
        the offset from x directly to the root."""
        if parent[x] == x:
            return x
        # Recursively find the root
        r = find(parent[x])
        # Accumulate the offset along the path
        val[x] = (val[x] + val[parent[x]]) % MOD
        parent[x] = r
        return r

    def get_val(x):
        """Return the total offset (expected wins) for node x."""
        r = find(x)
        return (val[x]) % MOD  # val[x] now is x->root offset; if x is root, it's the component offset

    def union(a, b, diff):
        """
        Attach root b to root a, and set val[b] so that
        for any node x in b's old component,
        get_val(x) increases by 'diff' relative to get_val(x) in a's component.
        Here 'diff' = add[b] - add[a] from the conceptual viewpoint.
        """
        # a,b must be roots when called
        parent[b] = a
        # we want: val[b] = diff, so that
        # for any x in old b-tree: get_val(x) = val[x] + val[b] + get_val(a)
        # which is old val[x] + diff + offset_of_a.
        val[b] = diff % MOD
        size[a] += size[b]

    # ------------------------------------------------------
    # Initialize DSU
    for i in range(1, N+1):
        parent[i] = i
        size[i] = 1
        val[i] = 0

    # ------------------------------------------------------
    # Process matches
    # - For match i, the team with p_i goes first, the team with q_i goes second
    # - a = size(first), b = size(second)
    # - Each member of first adds a/(a+b) to its expected-win count
    # - Each member of second adds b/(a+b) to its expected-win count
    # - Then the two teams merge
    #
    # We implement this by:
    #   or1 = find(p_i), or2 = find(q_i)
    #   inc1 = a * inv[a+b], inc2 = b * inv[a+b]
    #   val[or1] += inc1, val[or2] += inc2
    #   union by size => attach smaller to bigger
    #   if or1 is chosen as the final root, set the offset of the old second-root
    #     so that nodes in second get inc2 more than nodes in first.

    for i in range(N-1):
        r1 = find(p[i])
        r2 = find(q[i])
        if r1 == r2:
            # Should never happen as problem guarantees they're different teams.
            continue
        a = size[r1]
        b = size[r2]
        # Probability increments
        # inc1 = a/(a+b), inc2 = b/(a+b) in mod
        ab_inv = inv[a+b]  # 1/(a+b) mod
        inc1 = (a * ab_inv) % MOD
        inc2 = (b * ab_inv) % MOD

        # Add inc1 to entire component r1 => val[r1] += inc1
        # Add inc2 to entire component r2 => val[r2] += inc2
        # We'll do this by just adjusting val[r1] and val[r2],
        #   because get_val(x) for x in r1 is val[x] + val[r1->root].
        #   But first we must get the old offsets. Since r1, r2 are roots, get_val(r1)=val[r1], get_val(r2)=val[r2].
        # Now we set val[r1] = val[r1] + inc1, val[r2] = val[r2] + inc2

        val[r1] = (val[r1] + inc1) % MOD
        val[r2] = (val[r2] + inc2) % MOD

        # Now unify them by size
        # We want the bigger root to be the final root
        if size[r1] < size[r2]:
            # swap
            r1, r2 = r2, r1
            inc1, inc2 = inc2, inc1
        # Now r1 is bigger root
        # We want to attach r2 -> r1
        # We want that any node in r2 sees an offset of inc2 more than any node in r1
        # But we have already increased val[r2] by inc2, val[r1] by inc1.
        # The difference => (get_val(r2) - get_val(r1)) should be inc2 - inc1
        # Currently get_val(r2) = old_val(r2) + inc2
        #           get_val(r1) = old_val(r1) + inc1
        # The difference is (val[r2] - val[r1]) mod (since they are roots).
        # We want that difference to be (inc2 - inc1).
        # So "diff" = (get_val(r2) - get_val(r1)) - (inc2 - inc1), but we already
        # added inc2, inc1 to them. Actually we can set
        #   diff = ( val[r2] - val[r1] ) mod - ( inc2 - inc1 ) mod
        # But a simpler approach (matching the example) is:
        #   diff = val[r2] - val[r1]
        #   but we want that final difference = 0 + (inc2 - inc1).
        # So we do: needed = (inc2 - inc1) - (val[r2] - val[r1])

        curr_diff = (val[r2] - val[r1]) % MOD
        needed = ( (inc2 - inc1) % MOD - curr_diff ) % MOD

        # Now union(r1, r2, needed).
        parent[r2] = r1
        val[r2] = needed
        size[r1] += size[r2]

    # ------------------------------------------------------
    # Now compute final answers for each i by get_val(i)
    # Because get_val(i) = total expected wins for i, mod 998244353
    out = []
    for i in range(1, N+1):
        ans = get_val(i) % MOD
        out.append(str(ans))
    print(" ".join(out))