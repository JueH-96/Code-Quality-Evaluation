def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    X = int(next(it)) - 1  # convert X to 0-indexed
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(n)]
    # Read P and Q and convert them to 0-index (so each perm element is in 0..n-1).
    P = [int(next(it)) - 1 for _ in range(n)]
    Q = [int(next(it)) - 1 for _ in range(n)]
    
    # --- For red (P) ---
    # Build the cycle in P that contains X.
    red_cycle = []
    cur = X
    while True:
        red_cycle.append(cur)
        cur = P[cur]
        if cur == X:
            break
    red_component = set(red_cycle)
    # For every ball (except possibly at X) we must have its box in the cycle containing X.
    for i in range(n):
        if A[i] == 1 and i != X and i not in red_component:
            sys.stdout.write("-1")
            return

    # --- For blue (Q) ---
    blue_cycle = []
    cur = X
    while True:
        blue_cycle.append(cur)
        cur = Q[cur]
        if cur == X:
            break
    blue_component = set(blue_cycle)
    for i in range(n):
        if B[i] == 1 and i != X and i not in blue_component:
            sys.stdout.write("-1")
            return

    # --- Determine the minimal operations required ---
    # For the red channel, in the cycle [X, r1, r2, …, r_m],
    # if the first non-X box that initially contains a red ball is at index j (j>=1),
    # then every box from that index to the end must be cleared.
    red_set = set()
    j0 = None
    for j in range(1, len(red_cycle)):
        if A[red_cycle[j]] == 1:
            j0 = j
            break
    if j0 is not None:
        red_set = set(red_cycle[j0:])
    else:
        red_set = set()
    
    # Similarly for blue.
    blue_set = set()
    j0b = None
    for j in range(1, len(blue_cycle)):
        if B[blue_cycle[j]] == 1:
            j0b = j
            break
    if j0b is not None:
        blue_set = set(blue_cycle[j0b:])
    else:
        blue_set = set()
    
    # A single operation clears the ball(s) in one box.
    # So by performing operations on every box in the union of the needed boxes for red and blue,
    # we guarantee that eventually all balls will be moved into X.
    ans = len(red_set.union(blue_set))
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()