def solve():
    n = int(input())
    balls = []
    for _ in range(n):
        balls.append(list(map(int, input().split())))

    
    def check(subset):
        if not subset:
            return True
        
        for i in range(len(subset)):
            for j in range(len(subset)):
                if i == j:
                    continue
                if (balls[subset[i]][0] < balls[subset[j]][0] and balls[subset[i]][1] > balls[subset[j]][1]) or \
                   (balls[subset[i]][0] > balls[subset[j]][0] and balls[subset[i]][1] < balls[subset[j]][1]):
                    return False
        return True

    
    ans = set()
    for i in range(1 << n):
        subset = []
        for j in range(n):
            if (i >> j) & 1:
                subset.append(j)
        if check(subset):
            
            ans.add(tuple(sorted(subset)))

    print(len(ans))

solve()