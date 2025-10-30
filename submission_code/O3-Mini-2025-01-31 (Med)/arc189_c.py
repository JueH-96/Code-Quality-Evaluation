def main():
    import sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    X = int(next(it))
    # Read initial red ball presence (each 0 or 1)
    A = [int(next(it)) for _ in range(N)]
    # Read initial blue ball presence (each 0 or 1)
    B = [int(next(it)) for _ in range(N)]
    # Read the permutation P (for red) and Q (for blue)
    P = [int(next(it)) for _ in range(N)]
    Q = [int(next(it)) for _ in range(N)]

    # Explanation of the intended solution:
    #
    # Each operation works by choosing a box i (with some red and/or blue balls), clearing its content 
    # and moving its red balls to box P[i] and its blue balls to box Q[i]. In the final state we want
    # every box j ≠ X to be empty––i.e. all balls (from both colors) end up in box X.
    #
    # Since each ball “travels” by following a unique chain given by the permutation (either P for red or Q for blue),
    # an optimal strategy is to “clear” every box that ever ever holds a ball along the route from its starting box
    # (if the ball did not start in X) to X. (Remember: an operation does not remove a ball, but rather moves it
    # one step along its chain.) Moreover if two ball routes share some boxes, one operation on that common box
    # can serve both flows.
    #
    # Thus, if we define for a ball starting at box i (with i ≠ X) its route as the ordered sequence
    # of boxes visited from i (including i) until the first time X appears (X is not “cleared” by an operation),
    # then we must eventually perform an operation on each box that any ball visits.
    #
    # Therefore, the minimum number of operations equals the size of the union of all nodes (boxes) that lie on at least
    # one route coming from an initial ball (red and blue separately) – not counting X itself.
    #
    # There is one more necessary condition: if for a ball the chain never reaches X, that ball will circulate forever.
    # In our permutation setting this means that in the connected component (cycle) of the graph (either via P or Q),
    # the cycle must contain X if there is at least one ball starting in that component. Otherwise the goal is impossible.
    #
    # Thus, for each initial red ball (i with A[i]==1 and i != X), we follow the chain:
    #    i -> P[i] -> P[P[i]] -> … until we hit X.
    # While traversing, we record all the boxes that appear (we call this the "red route"). If a cycle is detected before hitting X,
    # then it is impossible.
    # We do the same for blue balls using permutation Q.
    #
    # Finally the answer is the number of distinct boxes (except for X) that are visited by at least one ball (either red or blue).
    
    # We use two arrays (for red and blue) that, for each box (indexed 1..N), mark whether an operation on that box is needed.
    red_union = [False]*(N+1)   # 1-indexed indexed; red_union[i]==True means box i is on the route from some red ball.
    blue_union = [False]*(N+1)  # similar definition for blue balls.
    red_processed = [False]*(N+1)   # used to memoize that we already know the chain from a given box is safe.
    blue_processed = [False]*(N+1)
    
    # Define a helper to process a chain starting from a given box "start"
    # following the mapping "mapping" (which is P for red and Q for blue).
    # The arrays 'processed' and 'union_arr' are used to mark nodes that have been confirmed
    # to be on a chain that eventually reaches X.
    # If along the chain (starting from "start") we find a cycle (i.e. we revisit a node) without meeting X,
    # then the function returns (False, temp_nodes) indicating failure.
    # Otherwise it returns (True, temp_nodes) and marks all nodes in the chain as processed (and included in union_arr).
    def process_chain(start, mapping, processed, union_arr):
        cur = start
        temp_nodes = []
        local_set = set()
        while True:
            if cur == X:
                break  # reached target; safe termination
            if processed[cur]:
                # Already know that the chain from this node eventually reached X.
                break
            if cur in local_set:
                # Cycle detected and X was not reached; impossible.
                return (False, temp_nodes)
            local_set.add(cur)
            temp_nodes.append(cur)
            cur = mapping[cur-1]  # move to the next box (permutation is 1-indexed)
        # Mark all nodes that we visited as processed (safe) and part of the union.
        for node in temp_nodes:
            processed[node] = True
            union_arr[node] = True
        return (True, temp_nodes)
    
    # Process red ball flows:
    for i in range(1, N+1):
        if A[i-1] == 1:
            # If a red ball is originally in X, no operation is needed for that ball.
            if i == X:
                continue
            if not red_processed[i]:
                ok, _ = process_chain(i, P, red_processed, red_union)
                if not ok:
                    sys.stdout.write("-1")
                    return

    # Process blue ball flows:
    for i in range(1, N+1):
        if B[i-1] == 1:
            if i == X:
                continue
            if not blue_processed[i]:
                ok, _ = process_chain(i, Q, blue_processed, blue_union)
                if not ok:
                    sys.stdout.write("-1")
                    return

    # The final answer is the number of boxes (except X) that were visited by at least one ball (red or blue).
    ans = 0
    for i in range(1, N+1):
        if i == X:
            continue
        if red_union[i] or blue_union[i]:
            ans += 1
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()