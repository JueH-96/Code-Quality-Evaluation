import sys

# Standard modular arithmetic functions
def power(base, exp, mod):
    res = 1
    base %= mod
    while exp > 0:
        if exp % 2 == 1:
            res = (res * base) % mod
        base = (base * base) % mod
        exp //= 2
    return res

def modInverse(n, mod):
    return power(n, mod - 2, mod)

def solve():
    A, B, M = map(int, sys.stdin.readline().split())

    N = A * B - 1

    # Precompute factorials modulo M up to N
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = (fact[i-1] * i) % M

    # Calculate product of hook lengths for the specific Young Diagram shape:
    # lambda_star = (A, A, ..., A, A-1)
    # This shape has (B-1) rows of length A, followed by 1 row of length A-1.
    
    prod_hook_lengths = 1

    # Iterate through cells (r, c) in the shape lambda_star
    # The shape consists of two main parts for hook length calculation:
    # 1. Cells in the first (B-1) rows: 1 <= r <= B-1
    #    For these rows, the row length is A.
    #    Column lengths: For 1 <= c <= A-1, column length is B.
    #                    For c = A, column length is B-1 (because the last row has length A-1)

    for r in range(1, B): # Iterate row index from 1 to B-1
        for c in range(1, A): # Iterate column index from 1 to A-1
            # Current cell is (r, c)
            # Cells to the right in current row: A - c
            # Cells below in current column: B - r (because this column goes all the way to row B)
            # Hook length h(r,c) = (A-c) + (B-r) + 1
            prod_hook_lengths = (prod_hook_lengths * (A - c + B - r + 1)) % M
        
        # Handle the cells in the A-th column for rows 1 to B-1
        # Current cell is (r, A)
        # Cells to the right in current row: A - A = 0
        # Cells below in current column: (B-1) - r  (since column A is one shorter, ending at row B-1)
        # Hook length h(r,A) = 0 + ((B-1)-r) + 1 = B - r
        prod_hook_lengths = (prod_hook_lengths * (B - r)) % M

    # 2. Cells in the last (B-th) row: r = B
    #    For this row, the row length is A-1.
    #    Column lengths: For 1 <= c <= A-1, column length is B.
    #    (Column A does not exist for r=B in this shape)

    for c in range(1, A): # Iterate column index from 1 to A-1 in the B-th row
        # Current cell is (B, c)
        # Cells to the right in current row: (A-1) - c
        # Cells below in current column: B - B = 0 (because this is the last row)
        # Hook length h(B,c) = ((A-1)-c) + 0 + 1 = A - c
        prod_hook_lengths = (prod_hook_lengths * (A - c)) % M

    # Calculate f_lambda_star using the hook length formula: N! / prod_hook_lengths
    # Note: N = AB - 1
    
    f_lambda_star = (fact[N] * modInverse(prod_hook_lengths, M)) % M

    # The final answer is (A-1) * (B-1) * f_lambda_star mod M
    # (A-1) and (B-1) are guaranteed to be >= 1 by constraints A, B >= 2
    ans = ((A - 1) * (B - 1)) % M
    ans = (ans * f_lambda_star) % M

    sys.stdout.write(str(ans) + '
')

solve()