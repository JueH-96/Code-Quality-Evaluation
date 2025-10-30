def solve():
    A, B, M = map(int, input().split())
    
    # Shape: B appears (A-1) times, then (B-1) once
    shape = [B] * (A - 1) + [B - 1]
    
    def hook_length_formula(shape):
        n = sum(shape)
        
        # Calculate hook lengths and their product
        hook_product = 1
        for i in range(len(shape)):
            for j in range(shape[i]):
                # Hook length = cells to right + cells below + 1
                hook_length = (shape[i] - j) + sum(1 for k in range(i + 1, len(shape)) if shape[k] > j)
                hook_product = (hook_product * hook_length) % M
        
        # Calculate n!
        factorial_n = 1
        for i in range(1, n + 1):
            factorial_n = (factorial_n * i) % M
        
        # Modular inverse using Fermat's little theorem (M is prime)
        def mod_inverse(a, m):
            return pow(a, m - 2, m)
        
        return (factorial_n * mod_inverse(hook_product, M)) % M
    
    return hook_length_formula(shape)

A, B, M = map(int, input().split())
print(solve())