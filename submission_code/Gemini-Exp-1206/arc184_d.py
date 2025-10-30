def solve():
    n = int(input())
    points = []
    for _ in range(n):
        points.append(list(map(int, input().split())))

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                if (points[i][0] < points[j][0] and points[i][1] < points[j][1]) or \
                   (points[i][0] > points[j][0] and points[i][1] > points[j][1]):
                    adj[j].append(i)

    ans = set()
    
    def backtrack(remaining, current_set):
        ans.add(tuple(sorted(current_set)))
        
        if not remaining:
            return

        for i in range(len(remaining)):
            next_remaining = []
            removed = set()
            
            for j in range(len(remaining)):
                if i != j:
                    if remaining[j] in adj[remaining[i]]:
                        removed.add(remaining[j])
                    
            for j in range(len(remaining)):
                if i != j and remaining[j] not in removed:
                    next_remaining.append(remaining[j])
            
            backtrack(next_remaining, current_set + [remaining[i]])

    backtrack(list(range(n)), [])
    print(len(ans) % 998244353)

solve()