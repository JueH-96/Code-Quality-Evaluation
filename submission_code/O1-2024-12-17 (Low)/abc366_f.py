def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parse inputs
    N = int(input_data[0])
    K = int(input_data[1])
    A = []
    B = []
    idx = 2
    for _ in range(N):
        Ai = int(input_data[idx]); Bi = int(input_data[idx+1])
        A.append(Ai)
        B.append(Bi)
        idx += 2

    # If K=1, a simple max over all f_i(1) suffices
    if K == 1:
        ans = max(A[i] + B[i] for i in range(N))
        print(ans)
        return

    # ----------------------------------------------------------------
    # We will use a "beam search" style approach because K <= 10, but N can be large.
    # 1) Select a candidate subset S of the functions, since Ai,Bi <= 50, we expect that
    #    the main contributors to a large final value will be those with relatively large
    #    A or B.  
    # 2) Then do a beam search of width M over sequences of length K from S, ensuring
    #    distinct indices.  Keep track of the maximum result.
    #
    # Candidate subset S construction:
    #   - pick top ~200 by A descending, then top ~200 by B descending, combine them
    #   - this keeps S reasonably small, on which we can do a beam search
    #
    # Beam search details:
    #   - State = (current_value, sequence_of_used_indices)
    #   - Start with (1, ()) because we begin evaluating from x=1
    #   - In each step, expand each state by applying any function in S not yet used,
    #     new_value = A[i] * old_value + B[i], new_sequence = old_sequence + (i)
    #   - Collect all next-states, sort by new_value descending, keep top M.
    #   - Repeat K times.  The best final new_value is our answer.
    #
    # M is chosen so that the search is feasible in Python.
    # ----------------------------------------------------------------

    # 1) Build an array of indices, then sort by A descending, B descending
    indices_by_A = sorted(range(N), key=lambda i: (A[i], B[i]), reverse=True)
    indices_by_B = sorted(range(N), key=lambda i: (B[i], A[i]), reverse=True)

    # Pick topX from each
    topX = 200  # can adjust if desired
    setA = set(indices_by_A[:topX])
    setB = set(indices_by_B[:topX])
    cand = list(setA.union(setB))
    # Just to be safe if N < topX
    cand = cand[:min(len(cand), 2*topX)]

    # We'll do a beam search with some beam width M
    M = 300  # beam width; may adjust for performance

    # We'll store states as (value, tuple_of_used_indices).
    # Start with the initial state: P=1, no used indices
    states = [(1, ())]

    import bisect

    for _ in range(K):
        next_states = []
        # Expand each state
        for (val, used) in states:
            used_set = set(used)  # for O(1) membership checks
            for i in cand:
                if i not in used_set:
                    new_val = A[i] * val + B[i]
                    # Build new "used" sequence
                    new_used = used + (i,)
                    # Collect
                    next_states.append((new_val, new_used))

        if not next_states:
            # No expansion possible (unlikely, but just in case)
            break

        # Keep only the top M by new_val
        # Sorting all of them by descending value
        next_states.sort(key=lambda x: x[0], reverse=True)
        if len(next_states) > M:
            next_states = next_states[:M]

        states = next_states

    # The answer is the best value among the final states
    ans = max(s[0] for s in states) if states else 1
    print(ans)

# Call main() as required
if __name__ == "__main__":
    main()