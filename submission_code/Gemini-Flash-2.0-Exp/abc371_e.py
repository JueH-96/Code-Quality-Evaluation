def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    total_distinct = 0
    for i in range(n):
        for j in range(i, n):
            subsequence = a[i:j+1]
            distinct_values = len(set(subsequence))
            total_distinct += distinct_values
    
    print(total_distinct)

solve()