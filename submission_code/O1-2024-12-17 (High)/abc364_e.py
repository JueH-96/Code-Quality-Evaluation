def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, X, Y = map(int, input_data[:3])
    dishes = [(0,0)]*N
    idx = 3
    for i in range(N):
        A = int(input_data[idx]); B = int(input_data[idx+1])
        dishes[i] = (A,B)
        idx += 2

    # Quick check: if all N dishes together never exceed (X,Y),
    # then Snuke can simply eat them all without ever stopping.
    # Because partial sums of nonnegative A_i,B_i cannot exceed the total.
    totalA = sum(d[0] for d in dishes)
    totalB = sum(d[1] for d in dishes)
    if totalA <= X and totalB <= Y:
        print(N)
        return

    # We know at most N dishes can be eaten.  
    # Key idea:
    #   - To eat k dishes, we only need the first (k-1) dishes' partial sum to be ≤(X,Y).
    #   - Then dish k itself may push sweetness or saltiness over the limit, but it is still eaten.
    #
    # Therefore, let "M" be the size of the largest subset one can form so that
    # the ENTIRE subset has sums of A and B each ≤ (X,Y).  (Then any prefix of it
    # also has partial sums ≤ (X,Y).)
    # If M = N, we can eat all N (since even the total never exceeds).
    # Otherwise, we can arrange those M dishes first (never exceeding (X,Y)) and then
    # pick one more dish (not in that subset) as the (M+1)-th dish, which is eaten
    # even if that final push exceeds the limit.
    # Also, even if M=0 (meaning no dish can fit entirely within (X,Y)), we can still
    # eat at least 1 dish by putting it first (partial sum=0 ≤(X,Y) before it).
    #
    # Implementation approach for finding M:
    #   - Disregard any dish whose (A> X and B> Y) altogether for *building* a within-(X,Y) subset
    #     because it can never fit inside partial sums ≤(X,Y).
    #   - Use a classic "knapsack-like" DP in one dimension (sum of A up to X),
    #     storing the minimal sum of B.  dp[k][a] = the minimal B-sum for some subset of size k
    #     with total A = a.  If dp[k][a] ≤ Y, that subset fits entirely in (X,Y).
    #   - We do this for all dishes that can possibly fit (i.e. A ≤ X, B ≤ Y).
    #   - Then M = max(k) for which there exists an a ≤ X with dp[k][a] ≤ Y.

    # Filter out dishes that can never be part of a fully-within-(X,Y) subset
    # i.e. items that individually exceed both X and Y won't help in building a subset
    # whose entire sum of A≤X and sum of B≤Y.  (They can still be eaten alone as the first dish,
    # so we do NOT discard them from the final answer logic, just from the "within" subset DP.)
    can_fit_dishes = []
    for (A,B) in dishes:
        # If a dish individually has A <= X and B <= Y, it *could* be part of
        # a subset that remains within (X,Y).  Otherwise, it cannot.
        if A <= X and B <= Y:
            can_fit_dishes.append((A,B))
    Mprime = len(can_fit_dishes)  # number of dishes that are candidates to remain fully within

    # Edge case: if no "can_fit" dishes at all, then M=0, but answer is at least 1
    # by placing any (possibly large) dish first.
    if Mprime == 0:
        # We can eat 1 dish by putting it first (partial-sum before it is 0≤(X,Y))
        print(1)
        return

    # Build DP array: dp[k][a] = minimal B-sum for picking exactly k dishes (from the ones
    # in can_fit_dishes) that sum up to A=a.  If none, dp[k][a] = some large "inf".
    INF = 10**15
    dp = [[INF]*(X+1) for _ in range(Mprime+1)]
    dp[0][0] = 0  # pick 0 dishes => sum(A)=0, sum(B)=0

    # Fill the DP by iterating over each candidate dish
    used_so_far = 0
    for (A,B) in can_fit_dishes:
        # For k from used_so_far down to 0, for a from X down to A
        # update dp[k+1][a].
        # This ensures each dish is used at most once.
        for k in range(used_so_far, -1, -1):
            # We'll go backwards in "a" to avoid reusing the same dish multiple times.
            # Only update a + A <= X.
            for a in range(X - A, -1, -1):
                if dp[k][a] == INF:
                    continue
                b_candidate = dp[k][a] + B
                if b_candidate < dp[k+1][a + A]:
                    dp[k+1][a + A] = b_candidate
        used_so_far += 1

    # Now dp is filled: dp[k][a] gives minimal B for a subset of size k (using those Mprime dishes)
    # that sums up A=a.  We want to find the largest k for which there is some a≤X with dp[k][a]≤Y.
    M = 0
    for k in range(Mprime+1):
        for a in range(X+1):
            if dp[k][a] <= Y:
                if k > M:
                    M = k

    # M is size of largest subset that remains entirely within (X,Y) among the can_fit_dishes.
    # Now relate M back to the full set of N.  If M == N, that means all dishes can be ordered
    # (or they all individually fit anyway) so that partial sums never exceed, so answer=N.
    # Otherwise, we can arrange those M dishes first and then pick 1 more dish from the leftover,
    # giving M+1 eaten (unless M=0, but then the answer is still 1).
    if M == N:
        print(N)
    else:
        # We can eat at least M+1 (or 1 if M=0).
        # But M+1 cannot exceed N, of course.  However, if M<N then M+1 ≤ N anyway.
        if M == 0:
            print(1)
        else:
            print(M+1)