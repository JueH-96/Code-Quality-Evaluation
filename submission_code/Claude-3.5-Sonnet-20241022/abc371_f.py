def solve():
    N = int(input())
    X = list(map(int, input().split()))
    Q = int(input())
    queries = []
    for _ in range(Q):
        t, g = map(int, input().split())
        queries.append((t-1, g))
    
    # Current positions of people
    pos = X[:]
    ans = 0
    
    for t, g in queries:
        # For each query, we need to move person t to position g
        target = t
        goal = g
        
        # If target person needs to move right
        if goal > pos[target]:
            # Check all people to the right that need to move
            to_move = []
            for i in range(N):
                if pos[i] > pos[target] and pos[i] <= goal:
                    to_move.append(i)
            
            # Move target and all blocking people to the right
            curr = pos[target]
            for p in to_move:
                dist = pos[p] - curr
                ans += dist
                pos[p] = curr + dist + 1
                curr = pos[p]
            
            # Finally move target person
            ans += goal - pos[target]
            pos[target] = goal
            
        # If target person needs to move left
        else:
            # Check all people to the left that need to move
            to_move = []
            for i in range(N):
                if pos[i] < pos[target] and pos[i] >= goal:
                    to_move.append(i)
            
            # Move target and all blocking people to the left
            curr = pos[target]
            for p in reversed(to_move):
                dist = curr - pos[p]
                ans += dist
                pos[p] = curr - (dist + 1)
                curr = pos[p]
            
            # Finally move target person
            ans += pos[target] - goal
            pos[target] = goal
    
    print(ans)

solve()