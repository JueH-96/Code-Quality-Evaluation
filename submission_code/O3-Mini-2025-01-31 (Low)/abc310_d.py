def main():
    import sys
    sys.setrecursionlimit(10**6)
    
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    T = int(next(it))
    M = int(next(it))
    
    # Record the incompatible pairs (convert to 0-indexed)
    incompat = []
    for _ in range(M):
        a = int(next(it)) - 1
        b = int(next(it)) - 1
        incompat.append((a, b))
    
    # We need to count the ways to partition players [0, 1, ..., N-1] into exactly T non-empty teams,
    # where the teams are considered as sets (unlabeled) and if any incompatible pair is on the same team,
    # that partition is invalid.
    #
    # Because N is small (≤ 10), we can generate all partitions by backtracking. The typical algorithm for
    # generating set-partitions in canonical order is to assign players one by one into existing teams or 
    # into a new team (only if the total number of teams so far is less than T).
    
    count = 0  # to count valid partitions
    
    # rec(i, groups): try to assign player i with the current partition in 'groups'.
    # groups is a list of lists where each inner list represents the players in that team.
    def rec(i, groups):
        nonlocal count
        # when all players are assigned, check if we have exactly T teams and then verify constraints.
        if i == N:
            if len(groups) == T:
                # Build a mapping from player to group index
                assign = [None] * N
                for idx, group in enumerate(groups):
                    for player in group:
                        assign[player] = idx
                valid = True
                for a, b in incompat:
                    if assign[a] == assign[b]:
                        valid = False
                        break
                if valid:
                    count += 1
            return
        
        # Try to assign player i to each existing team
        for j in range(len(groups)):
            groups[j].append(i)
            rec(i+1, groups)
            groups[j].pop()
        
        # Try to create a new team for player i, provided we haven't exceeded T teams.
        if len(groups) < T:
            groups.append([i])
            rec(i+1, groups)
            groups.pop()
    
    rec(0, [])
    sys.stdout.write(str(count))
    
if __name__ == "__main__":
    main()