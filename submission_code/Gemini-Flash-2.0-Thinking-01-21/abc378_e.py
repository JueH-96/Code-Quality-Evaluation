def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    prefix_sum = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix_sum[i] = prefix_sum[i-1] + a[i-1]
    
    total_sum = 0
    for l in range(1, n + 1):
        for r in range(l, n + 1):
            subarray_sum = prefix_sum[r] - prefix_sum[l-1]
            remainder = subarray_sum % m
            total_sum += remainder
            
    print(total_sum)

if __name__ == '__main__':
    solve()