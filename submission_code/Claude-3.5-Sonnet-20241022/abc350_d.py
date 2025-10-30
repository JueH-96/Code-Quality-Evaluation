def solve():
    N, M = map(int, input().split())
    
    # Create adjacency matrix to represent friendships
    friends = [[False]*N for _ in range(N)]
    
    # Mark initial friendships
    for _ in range(M):
        a, b = map(int, input().split())
        a -= 1
        b -= 1
        friends[a][b] = True
        friends[b][a] = True
    
    ans = 0
    
    # Keep trying to add new friendships until no more possible
    while True:
        found = False
        
        # Try all possible triplets
        for y in range(N):
            for x in range(N):
                if not friends[x][y]:
                    continue
                for z in range(N):
                    # Check if y-z are friends but x-z are not
                    if friends[y][z] and not friends[x][z]:
                        friends[x][z] = True
                        friends[z][x] = True
                        found = True
                        ans += 1
                        break
                if found:
                    break
            if found:
                break
                
        if not found:
            break
    
    print(ans)

solve()