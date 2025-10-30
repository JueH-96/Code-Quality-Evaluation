def solve():
    n, q = map(int, input().split())
    s = input()
    
    for _ in range(q):
        l, r = map(int, input().split())
        l -= 1
        count = 0
        for i in range(l, r - 1):
            if s[i] == s[i+1]:
                count += 1
        print(count)

solve()