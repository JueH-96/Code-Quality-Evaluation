def solve():
    n = int(input())
    a = list(map(int, input().split()))

    indexed_a = []
    for i in range(n):
        indexed_a.append((a[i], i + 1))
    
    indexed_a.sort(key=lambda x: x[0], reverse=True)
    
    print(indexed_a[1][1])

solve()