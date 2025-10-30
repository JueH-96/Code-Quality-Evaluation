def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    total_sum = 0
    for i in range(n):
        for j in range(i, n):
            subsequence = a[i:j+1]
            distinct_count = len(set(subsequence))
            total_sum += distinct_count
    print(total_sum)

solve()