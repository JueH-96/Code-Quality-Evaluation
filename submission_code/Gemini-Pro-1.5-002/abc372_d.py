# YOUR CODE HERE
def solve():
    n = int(input())
    h = list(map(int, input().split()))
    
    ans = []
    for i in range(n):
        count = 0
        for j in range(i + 1, n):
            max_height = 0
            for k in range(i + 1, j):
                max_height = max(max_height, h[k])
            if max_height < h[j]:
                count += 1
        ans.append(count)
    
    print(*ans)

solve()