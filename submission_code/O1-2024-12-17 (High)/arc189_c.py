def main():
    import sys
    input = sys.stdin.readline

    N, X = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    P = list(map(int, input().split()))
    Q = list(map(int, input().split()))

    # Convert X and the permutations to 0-based indexing
    X -= 1
    for i in range(N):
        P[i] -= 1
        Q[i] -= 1

    # Build inverse permutations of P and Q
    p_inv = [0]*N
    q_inv = [0]*N
    for i in range(N):
        p_inv[P[i]] = i
        q_inv[Q[i]] = i

    # Function to mark the cycle containing "start" in a boolean array "cycle"
    def mark_cycle(inv_perm, start):
        cycle = [False]*N
        cycle[start] = True
        cur = inv_perm[start]
        while cur != start:
            cycle[cur] = True
            cur = inv_perm[cur]
        return cycle

    # Mark the cycles in P and Q that contain X
    p_cycle = mark_cycle(p_inv, X)
    q_cycle = mark_cycle(q_inv, X)

    # Check possibility:
    #  Any red ball must lie in p_cycle (so it can reach X via P).
    #  Any blue ball must lie in q_cycle (so it can reach X via Q).
    for i in range(N):
        if A[i] == 1 and not p_cycle[i]:
            print(-1)
            return
        if B[i] == 1 and not q_cycle[i]:
            print(-1)
            return

    # Determine if we need to move any red ball from a box other than X
    # (i.e. does any box != X in p_cycle have a red ball?)
    has_red_outside = any(i != X and p_cycle[i] and A[i] == 1 for i in range(N))

    # Determine if we need to move any blue ball from a box other than X
    has_blue_outside = any(i != X and q_cycle[i] and B[i] == 1 for i in range(N))

    # If we do need to move red balls, in the simplest "push all to X" strategy,
    # we pick every node in p_cycle (except X) exactly once.
    # Similarly for blue balls in q_cycle (except X).
    # But if a node is in both cycles and we need to pick it for both colors,
    # one pick handles both. So effectively we need the size of the union of
    # (p_cycle\{X}) and (q_cycle\{X}) if that color indeed has balls outside X.

    # Count how many nodes we must pick (union of sets)
    # We'll do it by looping over all boxes and checking if this box belongs
    # to p_cycle (and we need red picks) or q_cycle (and we need blue picks).
    picks = 0
    for i in range(N):
        if i == X:
            continue
        need_red_here = p_cycle[i] and has_red_outside
        need_blue_here = q_cycle[i] and has_blue_outside
        if need_red_here or need_blue_here:
            picks += 1

    print(picks)

# Remember to call main() at the end
if __name__ == "__main__":
    main()