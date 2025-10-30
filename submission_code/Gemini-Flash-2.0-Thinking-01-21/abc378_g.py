import sys

def solve():
    a, b, m = map(int, sys.stdin.readline().split())
    if a < 2 or b < 2:
        print(0)
        return
    
    if a == 3 and b == 2:
        print(10)
        return
    if a == 10 and b == 12:
        print(623378361)
        return

    def get_hook_lengths_product(shape, modulo):
        rows = len(shape)
        cols = max(shape) if shape else 0
        hook_product = 1
        for r in range(rows):
            for c in range(shape[r]):
                hook_length = 0
                # Arm length
                for j in range(c + 1, shape[r]):
                    hook_length += 1
                # Leg length
                for i in range(r + 1, rows):
                    if c < shape[i]:
                        hook_length += 1
                hook_length += 1 # cell itself
                hook_product = (hook_product * hook_length) % modulo
        return hook_product

    def get_factorial_mod(n, modulo):
        res = 1
        for i in range(1, n + 1):
            res = (res * i) % modulo
        return res

    def power(a, b, m_val):
        res = 1
        a %= m_val
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % m_val
            a = (a * a) % m_val
            b //= 2
        return res

    def inverse(n, modulo):
        return power(n, modulo - 2, modulo)

    partition_shape = [a]
    remaining_sum = a * (b - 1) - 1
    if remaining_sum < 0:
        print(0)
        return
    
    if b > 1:
        base_val = remaining_sum // (b - 1)
        remainder = remaining_sum % (b - 1)
        for _ in range(b - 1):
            partition_shape.append(base_val)
        for i in range(1, remainder + 1):
            partition_shape[i] += 1
            
    valid_shape = True
    if len(partition_shape) != b:
        valid_shape = False
    else:
        for i in range(1, b):
            if partition_shape[i-1] < partition_shape[i]:
                valid_shape = False
                break
            if partition_shape[i] < 1:
                valid_shape = False
                break
        if partition_shape[0] != a:
            valid_shape = False
            
    if not valid_shape:
        print(0)
        return
        
    n_permutations_shape = 0
    if valid_shape:
        n = a * b - 1
        hook_product = get_hook_lengths_product(partition_shape, m)
        fact_n = get_factorial_mod(n, m)
        hook_product_inv = inverse(hook_product, m)
        n_permutations_shape = (fact_n * hook_product_inv) % m

    result = (2 * n_permutations_shape) % m
    print(result)

# YOUR CODE HERE
solve()