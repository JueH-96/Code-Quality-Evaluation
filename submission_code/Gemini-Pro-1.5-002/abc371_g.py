# YOUR CODE HERE
def solve():
    n = int(input())
    p = list(map(int, input().split()))
    a = list(map(int, input().split()))

    p = [x - 1 for x in p]
    a = [x - 1 for x in a]

    ans = list(a)

    current_a = list(a)
    
    for _ in range(n + 1):
        if current_a < ans:
            ans = list(current_a)
        
        next_a = [0] * n
        for i in range(n):
            next_a[i] = current_a[p[i]]
        current_a = next_a

    ans = [x + 1 for x in ans]
    print(*ans)

solve()