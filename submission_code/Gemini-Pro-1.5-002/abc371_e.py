# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    ans = 0
    for i in range(n):
        for j in range(i, n):
            subsequence = a[i:j+1]
            ans += len(set(subsequence))
    
    print(ans)

solve()