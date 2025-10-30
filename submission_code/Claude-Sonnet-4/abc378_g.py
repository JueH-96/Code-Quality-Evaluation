def solve():
    A, B, M = map(int, input().split())
    
    # We need to compute the number of standard Young tableaux
    # of rectangular shape (A-1) x (B-1)
    
    # Using hook-length formula
    # For rectangle (A-1) x (B-1), we have (A-1)*(B-1) boxes
    n = (A-1) * (B-1)
    
    # Calculate numerator: n!
    numerator = 1
    for i in range(1, n + 1):
        numerator = (numerator * i) % M
    
    # Calculate denominator: product of hook lengths
    denominator = 1
    for i in range(1, A):  # rows 1 to A-1
        for j in range(1, B):  # cols 1 to B-1
            hook_length = (A - i) + (B - j) - 1
            denominator = (denominator * hook_length) % M
    
    # Calculate modular inverse of denominator
    def mod_inverse(a, m):
        return pow(a, m - 2, m)  # Since M is prime, use Fermat's little theorem
    
    result = (numerator * mod_inverse(denominator, M)) % M
    print(result)

solve()