def f(x):
    # Keep dividing by 2 while x is even
    while x % 2 == 0:
        x //= 2
    return x

def solve():
    # Read input
    N = int(input())
    A = list(map(int, input().split()))
    
    # Calculate sum of f(A_i + A_j) for all i,j where i <= j
    total = 0
    for i in range(N):
        for j in range(i, N):
            sum_pair = A[i] + A[j]
            total += f(sum_pair)
    
    print(total)

# Run the solution
solve()