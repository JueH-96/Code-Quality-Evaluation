# YOUR CODE HERE
def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    result = 0
    xor_sum = [0] * (N + 1)
    
    for i in range(N):
        xor_sum[i + 1] = xor_sum[i] ^ A[i]
    
    for i in range(N):
        for j in range(i + 1, N):
            result += xor_sum[j + 1] ^ xor_sum[i]
    
    print(result)

solve()