import bisect

def solve():
    n, m, p = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    b_sorted = sorted(b)
    prefix_sums = [0] * (m + 1)
    for i in range(m):
        prefix_sums[i+1] = prefix_sums[i] + b_sorted[i]
    
    total_price = 0
    for i in range(n):
        p_prime = p - a[i]
        k_i = bisect.bisect_right(b_sorted, p_prime)
        contribution = a[i] * k_i + prefix_sums[k_i] + p * (m - k_i)
        total_price += contribution
        
    print(total_price)

if __name__ == '__main__':
    solve()