def main():
    import sys
    sys.setrecursionlimit(10000)
    input = sys.stdin.readline
    
    N, T, M = map(int, input().split())
    # incompat[i][j] = True if i and j cannot be in same team
    incompat = [[False]*N for _ in range(N)]
    for _ in range(M):
        a, b = map(int, input().split())
        a -= 1; b -= 1
        incompat[a][b] = True
        incompat[b][a] = True
    
    count = 0
    # block_members[b] = list of players in block b
    block_members = [[] for _ in range(T)]
    
    def dfs(player, num_blocks):
        nonlocal count
        if player == N:
            # all players assigned; accept only if we have exactly T non-empty blocks
            if num_blocks == T:
                count += 1
            return
        
        # Try to put 'player' into any existing non-empty block
        for b in range(num_blocks):
            ok = True
            for other in block_members[b]:
                if incompat[player][other]:
                    ok = False
                    break
            if not ok:
                continue
            block_members[b].append(player)
            dfs(player+1, num_blocks)
            block_members[b].pop()
        
        # Try to start a new block if we haven't used up T blocks
        if num_blocks < T:
            # new block index is num_blocks
            block_members[num_blocks].append(player)
            dfs(player+1, num_blocks+1)
            block_members[num_blocks].pop()
    
    # Kick off recursion
    dfs(0, 0)
    print(count)

if __name__ == "__main__":
    main()