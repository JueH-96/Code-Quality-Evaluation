def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast I/O handling
    # ----------------------------------------------------
    # The input format:
    # N
    # A_1 ... A_N
    # B_1 ... B_N
    # Q
    # query_1
    # ...
    # query_Q
    #
    # query can be:
    # 1 i x   (update A_i = x)
    # 2 i x   (update B_i = x)
    # 3 l r   (compute the maximum v after applying subrange l..r starting from v=0)
    #
    # Constraints (large), but here we provide a direct correct approach; it may not be optimal for the largest constraints.
    #
    # We'll implement the process exactly:
    #   v = 0
    #   for i in [l..r]:
    #       v = max(v + A[i], v * B[i])  (since all A[i], B[i] >= 1, and v≥0, the local-greedy condition is:
    #         if v*(B[i]-1) >= A[i], multiplication is bigger or equal, otherwise addition.)
    #
    # For each query of type 3, we do this step-by-step from l to r.

    # Parsing input
    idx = 0
    N = int(input_data[idx]); idx += 1
    A = list(map(int, input_data[idx : idx+N]))
    idx += N
    B = list(map(int, input_data[idx : idx+N]))
    idx += N
    Q = int(input_data[idx]); idx += 1

    out = []
    for _ in range(Q):
        t = int(input_data[idx]); idx += 1
        if t == 1:
            # 1 i x -> A[i-1] = x
            i = int(input_data[idx]); idx += 1
            x = int(input_data[idx]); idx += 1
            A[i-1] = x
        elif t == 2:
            # 2 i x -> B[i-1] = x
            i = int(input_data[idx]); idx += 1
            x = int(input_data[idx]); idx += 1
            B[i-1] = x
        else:
            # 3 l r -> compute
            l = int(input_data[idx]); idx += 1
            r = int(input_data[idx]); idx += 1
            l -= 1  # to 0-based
            r -= 1

            v = 0
            for i in range(l, r+1):
                # Decide whether to add or multiply
                # Equivalently check if v*(B[i] - 1) >= A[i] (and B[i]>1) => multiply
                # If B[i] == 1 => always addition
                if B[i] > 1 and v * (B[i] - 1) >= A[i]:
                    v = v * B[i]
                else:
                    v = v + A[i]
            out.append(str(v))

    print("
".join(out))

# Call solve() if this file is executed (as per the requirement).
def main():
    solve()

if __name__ == "__main__":
    main()