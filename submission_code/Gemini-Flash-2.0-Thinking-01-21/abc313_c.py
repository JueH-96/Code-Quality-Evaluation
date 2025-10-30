def solve():
    n = int(input())
    a = list(map(int, input().split()))
    if n <= 1:
        print(0)
        return
    
    s_0 = sum(a)
    q = s_0 // n
    r = s_0 % n
    
    sorted_a = sorted(a)
    
    cost = 0
    for i in range(n - r):
        cost += max(0, q - sorted_a[i])
    for i in range(n - r, n):
        cost += max(0, (q + 1) - sorted_a[i])
        
    print(cost)

if __name__ == '__main__':
    solve()