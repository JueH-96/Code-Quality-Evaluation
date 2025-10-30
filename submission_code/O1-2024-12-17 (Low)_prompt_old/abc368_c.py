def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # S(x) = sum of damage from attack 1 up through attack x.
    # Attack i does 1 damage if i % 3 != 0, else 3 damage.
    # Thus over each block of 3 attacks, total damage = 5.
    # So S(x) = 5 * (x // 3) + leftover(x % 3), where leftover(0) = 0, leftover(1) = 1, leftover(2) = 2.
    def S(x):
        return 5 * (x // 3) + (x % 3)

    # g(X) = smallest n >= 0 such that S(n) >= X.
    # We'll do a small search around n = 3*m where m = ceil(X/5).
    # This is guaranteed to be O(1) per call.
    def g(X):
        if X <= 0:
            return 0
        m = (X + 4) // 5  # ceil division for X/5
        # We search in a small window around 3*m to find the smallest n with S(n) >= X
        start = max(0, 3*m - 5)
        end = 3*m + 5
        best = end
        for n in range(start, end + 1):
            if S(n) >= X:
                best = n
                break
        return best

    T = 0  # total attacks made so far
    # For each enemy, we need to raise S(T) by at least H[i].
    # In other words, we want S(T + k) >= S(T) + H[i].
    # So define X = S(T) + H[i], and find the smallest n >= T s.t. S(n) >= X.
    # Then T = n.
    current_sum = 0  # This keeps S(T) without recomputing from scratch.
    # But we can compute S(T) quickly using S(T) function, so let's do that when needed.

    for h in H:
        X = S(T) + h  # we want S(new_T) >= X
        new_T = g(X)
        T = new_T

    print(T)