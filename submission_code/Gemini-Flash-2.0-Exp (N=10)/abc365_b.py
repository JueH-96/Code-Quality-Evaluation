def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    indexed_a = list(enumerate(a, 1))
    sorted_a = sorted(indexed_a, key=lambda x: x[1], reverse=True)
    
    print(sorted_a[1][0])

solve()