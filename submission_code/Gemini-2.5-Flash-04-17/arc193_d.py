import sys

def solve():
    N = int(sys.stdin.readline())
    A = sys.stdin.readline().strip()
    B = sys.stdin.readline().strip()

    a_pos = [i + 1 for i, char in enumerate(A) if char == '1']
    b_pos = [i + 1 for i, char in enumerate(B) if char == '1']

    m = len(a_pos)
    k = len(b_pos)

    if k > m:
        print(-1)
        return

    # The target configuration requires the set of occupied squares to be exactly
    # the set of indices j where B_j = 1.
    # Let the initial piece positions be a_0 < a_1 < ... < a_{m-1}.
    # After K operations, let the piece positions be p'_0 <= p'_1 <= ... <= p'_{m-1}.
    # The set of occupied squares is {p'_0, p'_1, ..., p'_{m-1}}.
    # This set must be equal to the target set of squares {b_0, b_1, ..., b_{k-1}},
    # where b_0 < b_1 < ... < b_{k-1} are the indices where B_j = 1.
    # This implies the sorted list of piece positions must be:
    # p'_j = b_j for j = 0, ..., k-1
    # p'_j = b_{k-1} for j = k, ..., m-1
    # The last m-k pieces must merge onto the last target square b_{k-1}.

    # A piece initially at a_j can reach position p'_j after K operations.
    # Each operation moves a piece at most 1 square in one direction.
    # If a piece is always moved to the right (p < i_t), its position increases by 1 each step.
    # If it's always moved to the left (p > i_t), its position decreases by 1 each step.
    # If it stays (p = i_t), its position stays.
    # The maximum possible final position for a piece starting at a_j after K steps is a_j + K.
    # So we must have p'_j <= a_j + K, which means K >= p'_j - a_j.
    # This must hold for all j = 0, ..., m-1.
    # Thus, K must be at least the maximum value of (p'_j - a_j) over all j.
    # Since K must be non-negative, K >= max(0, max_{j} (p'_j - a_j)).
    
    min_k = 0

    # Calculate max(p'_j - a_j) for j = 0, ..., m-1
    # For j = 0, ..., k-1: p'_j = b_pos[j]
    for j in range(k):
        min_k = max(min_k, b_pos[j] - a_pos[j])

    # For j = k, ..., m-1: p'_j = b_pos[k-1]
    # This loop only runs if k < m
    for j in range(k, m):
        min_k = max(min_k, b_pos[k-1] - a_pos[j])
        
    # The minimum number of operations is the maximum required shift, clamped at 0.
    # A positive required shift p'_j - a_j means the piece must move right net p'_j - a_j steps.
    # A negative required shift p'_j - a_j means the piece must move left net |p'_j - a_j| steps.
    # The maximum of p'_j - a_j determines the minimum K.
    
    # The crucial observation (from similar problems) is that the minimum K is indeed
    # max_{0 <= j < m} (p'_j - a_j) if that value is positive, and 0 otherwise.
    # This is because K operations allow any piece to shift its position by a value
    # between -K and +K, and specifically the difference (p'_j - a_j) is achievable
    # if K is large enough and the relative order is preserved. The transformation
    # on v = p+k values preserves the sorted order difference property which implies
    # the relative ordering of pieces based on their initial position is preserved
    # in the final configuration.

    print(min_k)


T = int(sys.stdin.readline())
for _ in range(T):
    solve()