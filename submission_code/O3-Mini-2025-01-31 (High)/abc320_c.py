def main():
    import sys

    data = sys.stdin.read().split()
    if not data:
        return
    M = int(data[0])
    reels = data[1:4]

    # Explanation:
    # For each reel, the symbols are arranged in a circle and the candidate stop times
    # that yield a given digit d come from the arithmetic progression:
    #    t = r + k*M   (with k>=0) for any index r such that the reel's symbol equals d.
    # Takahashi is allowed to press one button per second so the three stop times chosen
    # need to be distinct.
    # We want the minimum T (the maximum stop time among the three chosen times) such that
    # there exists a digit d and one candidate stopping time from each reel (from those available
    # for digit d and ≤ T) that are all distinct.
    #
    # Note that for each reel and each occurrence (position r in [0, M-1]) if S[i][r]==d then the
    # candidate times are r, r+M, r+2*M, ...
    # Since our goal is to minimize T (and r < M), even in the worst case we won’t need k > 2 because:
    #   Maximum candidate time = (M-1) + 2*M = 3*M - 1.
    #
    # So we iterate over T from 0 to 3*M and for each T and each digit d, we build a bipartite graph:
    #  • Left side: 3 reels (0,1,2)
    #  • Right side: time slots 0 to T.
    #  • Reel i connects to t if t is in its candidate set for digit d (i.e. t = r + k*M for some r with S[i][r]==d and t ≤ T).
    #
    # Then we check whether a matching of size 3 exists (each reel gets a distinct stopping time).
    #
    # If a matching exists for some digit d at time T, then T is a valid solution.
    # We output the smallest such T. If none is found, we output -1.
    
    maxT = 3 * M  # We need to consider T up to 3*M (T goes from 0 to 3*M)
    
    # Iterate over T (the maximum stopping time) from 0 to maxT.
    for T in range(maxT + 1):
        # Try for every possible digit from '0' to '9'
        for d in map(str, range(10)):
            # Build the list of candidate times for each reel for this digit d.
            graph = []  # graph[i] will store the candidate times (t) for reel i.
            valid = True
            for i in range(3):
                cand = []
                for r in range(M):
                    if reels[i][r] == d:
                        # For each candidate time from this occurrence r,
                        # try k = 0, 1, 2; t = r + k*M.
                        for k in range(3):
                            t_val = r + k * M
                            if t_val <= T:
                                cand.append(t_val)
                if not cand:
                    valid = False
                    break
                graph.append(cand)
            if not valid:
                continue  # This digit cannot be produced on one of the reels up to time T.
            
            # Now, we have a bipartite graph: left side nodes (reels 0, 1, 2)
            # and right side nodes (time slots 0,1,...,T). There is an edge from reel i
            # to time slot t if t is in graph[i]. We want a matching that covers all 3 reels.
            match = [-1] * (T + 1)  # match[t] = reel assigned to time slot t, or -1 if none.

            # Standard DFS based bipartite matching.
            def dfs(reel, used):
                for t in graph[reel]:
                    if used[t]:
                        continue
                    used[t] = True
                    if match[t] == -1 or dfs(match[t], used):
                        match[t] = reel
                        return True
                return False

            match_count = 0
            for i in range(3):
                used = [False] * (T + 1)
                if dfs(i, used):
                    match_count += 1
            
            # If we can assign distinct times for all three reels, then T is a solution.
            if match_count == 3:
                sys.stdout.write(str(T))
                return

    sys.stdout.write("-1")

if __name__ == '__main__':
    main()