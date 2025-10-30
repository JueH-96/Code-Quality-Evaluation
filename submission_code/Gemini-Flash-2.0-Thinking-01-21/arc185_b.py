def solve():
    n = int(input())
    a = list(map(int, input().split()))
    total_sum = sum(a)
    q = total_sum // n
    r = total_sum % n
    
    def get_target_prefix_sum(k):
        if k <= n - r:
            return k * q
        else:
            return k * q + r - (n - k)
            
    current_prefix_sum = 0
    for k in range(1, n):
        current_prefix_sum += a[k-1]
        target_prefix_sum = get_target_prefix_sum(k)
        if current_prefix_sum > target_prefix_sum:
            print("No")
            return
            
    print("Yes")

t = int(input())
for _ in range(t):
    solve()