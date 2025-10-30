def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    mod = 998244353

    N = int(input_data[0])
    M = int(input_data[1])

    # Read constraints
    constraints = []
    ptr = 2
    
    # We will also keep track of how many times each position x appears as X_i
    # (for any still-active constraint).  Those positions are "forbidden"
    # to take the current largest value, because if the largest value lands
    # on X_i for a constraint that includes that position, it would violate
    # "maximum in [L_i .. R_i] is not P_{X_i}".    
    freqX = [0]*(N+1)  # 1-based indexing: freqX[x] = how many active constraints have X_i = x

    # Quick check: if L_i == R_i == X_i for any constraint => impossible 
    # (that subarray's max is forced to be P_{X_i}, contradicting the constraint).
    # In that case the answer is 0 immediately.
    for i in range(M):
        L = int(input_data[ptr]); R = int(input_data[ptr+1]); X = int(input_data[ptr+2])
        ptr += 3
        if L == R == X:
            print(0)
            return
        constraints.append((L, R, X))

    # For quickly "removing" (satisfying) constraints once we place a large value
    # in some position p, we need to know which constraints are covered by p
    # (i.e. p in [L..R]).  We'll store for each position p the list of constraints
    # that have L_i <= p <= R_i.
    #
    # Naively, we might do:
    #   for each constraint i with L_i..R_i
    #       for p in L_i..R_i:
    #           constraints_of_pos[p].append(i)
    #
    # That can be up to N*M = 500*10^5 = 5e7 operations, which is borderline
    # but often still doable in optimized C++. In Python we must be careful.
    #
    # We will implement it carefully; if it is too slow in practice,
    # one could do more sophisticated "sweep line" but we'll try the direct way
    # with fast methods.

    # Initialize adjacency
    constraints_of_pos = [[] for _ in range(N+1)]  # constraints_of_pos[p] = list of constraint indices i

    # Also initialize freqX
    for i,(L,R,X) in enumerate(constraints):
        freqX[X] += 1

    # Build the adjacency (this is the heavy part).
    for i,(L,R,X) in enumerate(constraints):
        for p in range(L, R+1):
            constraints_of_pos[p].append(i)

    # "active[i]" will tell if a constraint i is still not satisfied.
    # Once we place a large number in one of its [L_i..R_i], it's satisfied.
    active = [True]*M

    # "used[p]" tells if position p already took one of the larger values
    used = [False]*(N+1)

    ways = 1  # our result

    # We go from the largest value N down to 1.
    # At each step v, we look at how many positions are "forbidden" (i.e. freqX[x]>0)
    # among those that are not used.  Let f = that count.
    # We have v positions left unused, so the number of free positions is c = v - f.
    # If c <= 0, answer=0.  Otherwise we multiply ways by c.
    #
    # Then to keep the logic consistent, we "simulate" picking exactly one
    # of those free positions p to place the value v.  We mark used[p]=True,
    # and then "remove" all constraints that p satisfies => those constraints
    # become inactive, and so freqX[X_i]-- for each such constraint.

    free_count = N  # how many positions are not used so far

    for v in range(N, 0, -1):
        # Count how many are forbidden among the unused positions
        # i.e. x in [1..N], not used, freqX[x] > 0
        forbidden_count = 0
        # We only need to know the final number, so a loop over all positions:
        # N=500 => looping is fine
        for x in range(1, N+1):
            if not used[x] and freqX[x] > 0:
                forbidden_count += 1

        # c = number of free positions
        c = v - forbidden_count  # because we have v not-used positions, f are forbidden
        if c <= 0:
            ways = 0
            break
        ways = (ways * c) % mod

        # Now we pick an actual free position p (any one) to place v.
        # We'll do: pick the smallest x s.t. used[x]=False and freqX[x]=0
        # (just to keep a consistent logic).
        # Then we remove the constraints that p satisfies.
        p_chosen = -1
        for x in range(1, N+1):
            if (not used[x]) and freqX[x] == 0:
                p_chosen = x
                break

        used[p_chosen] = True
        free_count -= 1

        # Remove constraints satisfied by placing v in p_chosen
        for ci in constraints_of_pos[p_chosen]:
            if active[ci]:
                active[ci] = False
                # That constraint has X_i => reduce freqX[X_i]
                _, _, Xc = constraints[ci]
                freqX[Xc] -= 1

    print(ways % mod)

# Don't forget to call main()
if __name__ == "__main__":
    main()