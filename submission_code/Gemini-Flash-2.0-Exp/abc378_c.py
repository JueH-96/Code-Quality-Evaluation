def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    b = []
    for i in range(n):
        found = False
        last_occurrence = -1
        for j in range(i):
            if a[i] == a[j]:
                found = True
                last_occurrence = j + 1
        b.append(last_occurrence)
    
    print(*b)

solve()