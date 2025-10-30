def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    pos = 1
    out = []
    
    # Because for any K >= 1 (and K < N), the array indices form a single
    # connected component (you can propagate values throughout by stepping
    # at distance <= K repeatedly), the key constraint is that you cannot
    # "create" a new value that was not already present somewhere in A.
    #
    # Additionally, to end up with a particular value v in the final array,
    # you must NOT overwrite all occurrences of v in A while trying to fix
    # other positions. In particular, if you need v in B, you must keep at
    # least one copy of v in A that ends up not being overwritten.
    #
    # Practically, if for value v needed by B, there is no position i where
    # A_i == B_i == v (i.e. a place that can remain v without needing a change),
    # then all places that have v in A are forced to be overwritten to match B.
    # This destroys v completely (since once all original v's are overwritten,
    # you can no longer replicate v). Hence you cannot produce v for the
    # positions in B that need it. The answer in that case is "No."
    #
    # Otherwise, for every v in B, there exists at least one i where A_i = B_i = v.
    # That index can be left unchanged as a "source" of v, allowing
    # the replication of v to all needed positions (because the entire array
    # is connected). The answer then is "Yes."

    # Implementation steps for each test:
    # 1) Read N, K, A, B
    # 2) Collect distinct values in B into needed set
    # 3) Scan indices to find where A_i == B_i; mark found[A_i] = True
    # 4) If for every v in needed, found[v] is True, print Yes; else No.

    idx = 0
    for _ in range(t):
        N = int(input_data[pos]); pos += 1
        K = int(input_data[pos]); pos += 1  # K is unused directly, but read it
        A = input_data[pos:pos+N]
        pos += N
        B = input_data[pos:pos+N]
        pos += N
        
        # Convert A, B to integers (they are up to N, which is safe)
        A = list(map(int, A))
        B = list(map(int, B))
        
        needed = set(B)
        found = {}
        
        for v in needed:
            found[v] = False
        
        for i in range(N):
            if A[i] == B[i]:
                val = A[i]
                if val in needed:
                    found[val] = True
        
        # Check if all needed values are found
        can_do = True
        for v in needed:
            if not found[v]:
                can_do = False
                break
        
        if can_do:
            out.append("Yes")
        else:
            out.append("No")

    print("
".join(out))

def main():
    solve()

# Call solve() according to the requested format
if __name__ == "__main__":
    main()