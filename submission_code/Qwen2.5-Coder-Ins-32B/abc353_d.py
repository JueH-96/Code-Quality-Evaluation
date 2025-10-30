# YOUR CODE HERE
MOD = 998244353

def f(x, y):
    return int(str(x) + str(y))

def solve():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:]))
    
    # Calculate prefix sums of lengths of string representations
    len_prefix = [0] * (N + 1)
    for i in range(N):
        len_prefix[i + 1] = len_prefix[i] + len(str(A[i]))
    
    # Calculate the result
    result = 0
    for i in range(N):
        for j in range(i + 1, N):
            result += f(A[i], A[j])
            result %= MOD
    
    print(result)

solve()