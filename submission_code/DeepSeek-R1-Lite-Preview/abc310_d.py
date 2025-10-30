def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    M = int(data[2])
    
    adj = [set() for _ in range(N)]
    index = 3
    for _ in range(M):
        A = int(data[index]) - 1
        B = int(data[index + 1]) - 1
        adj[A].add(B)
        adj[B].add(A)
        index += 2
    
    # Order players by decreasing degree
    players = sorted(range(N), key=lambda x: len(adj[x]), reverse=True)
    
    count = 0
    color = [0] * N  # color[i] is the team assigned to player i

    def backtrack(pos, used_colors):
        nonlocal count
        if pos == N:
            if used_colors == (1 << T) - 1:
                count += 1
            return
        player = players[pos]
        for team in range(1, T + 1):
            # Check if this team is compatible with all incompatible players
            compatible = True
            for neighbor in adj[player]:
                if color[neighbor] == team:
                    compatible = False
                    break
            if compatible:
                color[player] = team
                used_colors |= (1 << (team - 1))
                backtrack(pos + 1, used_colors)
                color[player] = 0
                used_colors ^= (1 << (team - 1))
    
    backtrack(0, 0)
    print(count)

if __name__ == '__main__':
    main()