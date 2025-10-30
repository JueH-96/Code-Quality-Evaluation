def solve(N):
    # Remove all factors of 2
    x = 0
    while N % 2 == 0:
        N //= 2
        x += 1
    
    # Remove all factors of 3
    y = 0
    while N % 3 == 0:
        N //= 3
        y += 1
    
    # If anything remains, it's not of form 2^x * 3^y
    return N == 1

N = int(input())
print("Yes" if solve(N) else "No")