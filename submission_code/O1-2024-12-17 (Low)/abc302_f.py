def main():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    # Parse N, M
    N, M = map(int, input_data[:2])

    # Read the sets S_i
    # We'll keep:
    #  - S: list of sets (or lists) of elements
    #  - pos[e]: list of indices of sets that contain element e
    S = []
    pos = [[] for _ in range(M+1)]  # pos[e] will hold indices of sets that contain element e

    idx = 2
    for i in range(N):
        A_i = int(input_data[idx])
        idx += 1
        elements = list(map(int, input_data[idx:idx+A_i]))
        idx += A_i
        S.append(elements)
        for e in elements:
            pos[e].append(i)

    # We want to see if it's possible to form a set with 1 and M.
    # If any set already contains both 1 and M, answer is 0.
    # Otherwise we do a BFS over "sets" where edges exist if they share an element.
    # BFS steps:
    # 1) Start queue with all sets that contain 1 at distance 0.
    # 2) For a set X, if it contains M, return dist[X].
    # 3) For each element e in X, if we haven't 'expanded' e yet,
    #       mark element e as expanded, then for each set Y in pos[e],
    #       if not visited, set dist[Y] = dist[X]+1, push Y.

    # If no set contains 1, it's immediately impossible
    # If BFS completes without finding M, answer -1.

    # Find all sets that contain 1
    starts = pos[1]
    if not starts:
        # no set even contains 1
        print(-1)
        return

    # We'll do a quick check if any set already has both 1 and M
    for st in starts:
        # st is an index of a set that contains 1
        # check if M in that set
        # S[st] is the list of elements in set st
        # but let's do a membership check
        # We'll do it as a set for faster membership (for large sets).
        # Or we can do a linear check if needed; either is fine.
        # Summation of A_i is large, but a one-time set creation might be okay.
        # We'll just do "if M in S[st]" in Python. It's O(A_i), but only for starts sets, can be large but typically less than union BFS cost.
        # But let's do the BFS approach anyway; we need BFS if not found in these sets.
        pass

    # Build a quick membership check for every set, only once, if we need it:
    # We'll skip building separate membership structures if we can do a quick search now;
    # If that fails, BFS anyway. This one-time cost across all sets might be too high.
    # Instead, let's do a quick check for each st if M is in S[st] by a direct "if M in S[st]".
    # That might be fine as a worst-case fallback.
    for st in starts:
        if M in S[st]:
            print(0)
            return

    # BFS
    visited_set = [False]*N
    dist = [-1]*N
    from collections import deque

    queue = deque()

    # Initialize queue with sets that contain 1
    for st in starts:
        visited_set[st] = True
        dist[st] = 0
        queue.append(st)

    visited_element = [False]*(M+1)  # to avoid re-expanding adjacency on the same element

    while queue:
        cur_set_idx = queue.popleft()
        dcur = dist[cur_set_idx]

        # Check if this set contains M
        # We could do "if M in S[cur_set_idx]" but let's do membership check:
        # We'll do a linear scan (S[cur_set_idx] can be large but BFS must also handle them).
        # If it is found, return dcur.
        # Actually, we only need to check right when we pop the set from the queue.
        # Because if it has M, we're done.
        # But we already checked if M was in sets that contain 1. We should check for others too.
        # The BFS might lead us to new sets. So let's do a quick membership check again.
        for e in S[cur_set_idx]:
            if e == M:
                print(dcur)
                return

        # Expand adjacency
        for e in S[cur_set_idx]:
            if not visited_element[e]:
                visited_element[e] = True
                for nxt_set_idx in pos[e]:
                    if not visited_set[nxt_set_idx]:
                        visited_set[nxt_set_idx] = True
                        dist[nxt_set_idx] = dcur + 1
                        queue.append(nxt_set_idx)

    # If we exit BFS without finding a set that contains M, it's impossible
    print(-1)

# Don't forget to call main()
if __name__ == "__main__":
    main()