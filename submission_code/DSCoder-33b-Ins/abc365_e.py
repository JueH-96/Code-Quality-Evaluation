# YOUR CODE HERE

N = int(input())
A = list(map(int, input().split()))

def xor_sum(A):
    xor_sum = 0
    for i in range(len(A)):
        xor_val = 0
        for j in range(i, len(A)):
            xor_val ^= A[j]
            xor_sum += xor_val
    return xor_sum

print(xor_sum(A))