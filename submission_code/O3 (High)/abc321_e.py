import sys

# ---------- auxiliary ---------- #
def count_desc(node: int, depth: int, N: int) -> int:
    """
    How many vertices are exactly `depth` edges below `node`
    and do not exceed N ?
    depth = 0  -> only node itself (if node<=N)
    """
    if depth < 0 or node > N:
        return 0

    # the smallest power of two that is >  N//node  tells us the
    # deepest complete level that can still fit into the range 1..N
    max_possible_depth = (N // node).bit_length() - 1   # floor(log2(N//node))
    if depth > max_possible_depth:                      # whole layer lies outside [1,N]
        return 0

    width = 1 << depth                                  # 2**depth
    left = node * width                                 # left-most index of that layer
    right = left + width - 1                            # right-most index (in the infinite tree)

    if left > N:                                        # no vertex survives the truncation
        return 0
    return min(right, N) - left + 1                     # vertices that are ≤ N


# ---------- single test case ---------- #
def solve_one(N: int, X: int, K: int) -> int:
    ans = 0

    depth_up = 0            # distance from X to the current ancestor
    curr = X                # current ancestor
    child_on_path = None    # child of 'curr' that lies on the path towards X

    while True:             # walk from X upwards until we reach the root (1)
        rem = K - depth_up  # edges still available for the downward part

        if rem == 0:                        # the ancestor itself is at distance K
            ans += 1
        elif rem > 0:
            if depth_up == 0:               # still at X → both children allowed
                ans += count_desc(curr, rem, N)
            else:                           # only the sibling branch is allowed
                sibling = child_on_path ^ 1         # flip least significant bit
                if sibling <= N:
                    ans += count_desc(sibling, rem - 1, N)

        if curr == 1:                       # reached the root
            break

        child_on_path = curr                # next iteration: this is the child on path
        curr //= 2                          # move to the parent
        depth_up += 1

    return ans


# ---------- main ---------- #
def main() -> None:
    data = sys.stdin.buffer.read().split()
    t = int(data[0])
    out = []
    p = 1
    for _ in range(t):
        N = int(data[p]);   X = int(data[p+1]);   K = int(data[p+2])
        p += 3
        out.append(str(solve_one(N, X, K)))
    sys.stdout.write("
".join(out))

# run
if __name__ == "__main__":
    main()