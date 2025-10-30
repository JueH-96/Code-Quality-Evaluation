def solve():
    n, k = map(int, input().split())
    s = input()
    
    count = 0
    i = 0
    while i < n:
        if s[i] == 'B':
            count += 1
            i += k
            while i < n and s[i-1] == 'B':
                i += 0
                break
        else:
            i += 1
    print(count)

t = int(input())
for _ in range(t):
    solve()