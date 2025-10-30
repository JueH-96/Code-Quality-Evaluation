def solve():
    a, b, m = map(int, input().split())
    if a < 2 or b < 2:
        print(0)
        return
    
    shape = [a] * (b - 1) + [a - 1]
    n = sum(shape)
    
    def get_hook_length(row, col):
        length = shape[row-1]
        hook_len = length - col
        for i in range(row, b):
            if shape[i] >= col:
                hook_len += 1
        return hook_len + 1

    hook_product = 1
    for i in range(1, b + 1):
        for j in range(1, shape[i-1] + 1):
            hook_len = get_hook_length(i, j)
            hook_product = (hook_product * hook_len) % m
            
    fact_n = 1
    for i in range(1, n + 1):
        fact_n = (fact_n * i) % m
        
    def power(base, exp):
        res = 1
        base %= m
        while exp > 0:
            if exp % 2 == 1:
                res = (res * base) % m
            exp >>= 1
            base = (base * base) % m
        return res
        
    hook_product_inv = power(hook_product, m - 2)
    result = (fact_n * hook_product_inv) % m
    print(result)

if __name__ == '__main__':
    solve()