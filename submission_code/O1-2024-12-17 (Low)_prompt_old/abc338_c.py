def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    Q = list(map(int, data[1:1+N]))
    A = list(map(int, data[1+N:1+2*N]))
    B = list(map(int, data[1+2*N:1+3*N]))

    # If it's impossible to make any dish (check quickly: if for some i both A_i and B_i > 0 but Q_i == 0, or if for all i with A_i>0, Q_i==0, etc.)
    # a simpler check is done by the loop below anyway; if no combination is feasible, answer will be 0.

    # Maximum x (servings of dish A) is limited by the minimum Q_i//A_i for i with A_i>0
    # (because you can't exceed that or you'll run out of that ingredient).
    max_x = float('inf')
    for i in range(N):
        if A[i] > 0:
            max_x = min(max_x, Q[i] // A[i])
    if max_x == float('inf'):  # Means A is all zeros, but problem guarantees at least one A_i>=1
        print(0)
        return

    best = 0

    for x in range(max_x + 1):
        # Check feasibility and compute leftover for each ingredient:
        # Then see how many servings of B we can make with that leftover
        feasible = True
        max_y = float('inf')
        for i in range(N):
            needed_A = A[i]*x
            if needed_A > Q[i]:
                feasible = False
                break
            leftover = Q[i] - needed_A
            if B[i] > 0:
                max_y = min(max_y, leftover // B[i])

        if feasible:
            # If max_y is still inf, that means all B_i's are 0, which can't happen since
            # there's at least one B_i >= 1 by problem statement, so max_y should be some integer
            y = max_y if max_y != float('inf') else 0
            if x + y > best:
                best = x + y

    print(best)

# Let's call solve() to execute
# (not required in some environments, but as per the specification, we do call it)
solve()