def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def solve():
    n, m, c, k_val = map(int, input().split())
    a = list(map(int, input().split()))
    
    if k_val == 0:
        print(0)
        return
        
    period = m // gcd(c, m)
    
    total_sum = 0
    
    for iteration in range(k_val):
        min_val = m
        for i in range(n):
            current_val = (c * iteration + a[i]) % m
            min_val = min(min_val, current_val)
        total_sum += min_val
        
    print(total_sum)

if __name__ == '__main__':
    solve()