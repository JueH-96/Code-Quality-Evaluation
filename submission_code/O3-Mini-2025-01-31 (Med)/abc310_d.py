def main():
    import sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    T = int(next(it))
    M = int(next(it))
    
    # Build the conflict matrix for players [0, N-1]
    conflict = [[False]*N for _ in range(N)]
    for _ in range(M):
        a = int(next(it)) - 1  # Convert 1-indexed to 0-indexed.
        b = int(next(it)) - 1
        conflict[a][b] = True
        conflict[b][a] = True

    # We'll enumerate all partitions (unlabeled teams) of players into exactly T teams.
    # In each partition, every player belongs to exactly one team and each team is non-empty.
    # Two divisions are different when there exists a pair of players that are in the same team in one partition
    # and in different teams in the other.
    # We use recursion that follows a canonical order (restricted growth function) to ensure unique partitions.
    
    count = 0
    groups = []  # groups is a list of current teams, each team is represented as a list of player indices.

    def backtrack(i):
        nonlocal count
        if i == N:
            # Only count partitions with exactly T teams.
            if len(groups) == T:
                count += 1
            return

        # Try to assign player i to any existing team where conflicts are avoided.
        for j in range(len(groups)):
            valid = True
            for player in groups[j]:
                # If conflict exists between player i and a player in the same team, skip assignment.
                if conflict[i][player]:
                    valid = False
                    break
            if valid:
                groups[j].append(i)
                backtrack(i+1)
                groups[j].pop()
        # Option to start a new team if we haven't reached T teams yet.
        if len(groups) < T:
            groups.append([i])
            backtrack(i+1)
            groups.pop()
    
    backtrack(0)
    
    sys.stdout.write(str(count))

if __name__ == '__main__':
    main()