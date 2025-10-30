# YOUR CODE HERE
import sys

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = []
    index = 2
    for _ in range(N):
        A.append(list(map(int, data[index:index+M])))
        index += M
    
    # Precompute the transformation steps
    # We need to find the minimal x such that after x steps, A_i and A_j are the same
    # The transformation is: for each step, A_i[k] = (sum_{l=1}^k A_i[l]) mod 2
    
    # To find the minimal x, we can precompute the transformation steps for each sequence
    # and then find the minimal x such that the transformed sequences are the same
    
    # The transformation is linear and can be represented as a matrix multiplication
    # However, it's easier to precompute the transformed sequences for all possible x up to a certain limit
    
    # Since the transformation is periodic with period 2^M, we can limit x to 2^M
    # But 2^M can be up to 2^20, which is 1e6, which is manageable
    
    # So, for each sequence, we precompute the transformed sequences for x = 0, 1, ..., 2^M - 1
    # Then, for each pair (i,j), we find the minimal x such that the transformed sequences are the same
    
    # However, precomputing for all x up to 2^M is not feasible for M=20 (2^20 is 1e6)
    # So, we need a smarter approach
    
    # Let's consider the transformation as a linear transformation over GF(2)
    # The transformation can be represented as a matrix T, where T^k is the transformation after k steps
    
    # The minimal x such that T^x A_i = T^x A_j is the minimal x such that T^x (A_i - A_j) = 0
    
    # The order of T is the smallest x such that T^x = I (identity matrix)
    # The order of T is 2^M, since the transformation is a shift register with period 2^M
    
    # So, the minimal x is the smallest x such that T^x (A_i - A_j) = 0
    
    # To find this x, we can represent A_i - A_j as a vector in GF(2)^M, and find the smallest x such that T^x v = 0
    
    # The transformation T is a lower triangular matrix with 1s on the diagonal and below
    # T^k is also a lower triangular matrix with 1s on the diagonal and below, but the pattern is more complex
    
    # Instead of computing T^x, we can simulate the transformation step by step
    
    # So, for each pair (i,j), we compute the difference vector v = A_i - A_j (element-wise XOR)
    # Then, we apply the transformation step by step until v becomes zero, and count the number of steps
    
    # Since the transformation is periodic with period 2^M, we can limit the number of steps to 2^M
    
    # Now, let's implement this
    
    # Precompute the transformation steps for each sequence
    # For each sequence, we can precompute the transformed sequences for x = 0, 1, ..., 2^M - 1
    # But this is not feasible for large M
    
    # Instead, for each pair (i,j), we compute the difference vector and simulate the transformation until it becomes zero
    
    # So, for each pair (i,j), we do the following:
    # 1. Compute the difference vector v = A_i XOR A_j
    # 2. Apply the transformation step by step until v becomes zero, and count the number of steps
    
    # The transformation step is: for each k, v[k] = (sum_{l=1}^k v[l]) mod 2
    
    # Now, let's implement this
    
    # First, precompute the transformation steps for all sequences
    # Since the transformation is linear, we can represent it as a matrix multiplication
    # But it's easier to simulate the transformation step by step
    
    # So, for each sequence, we can precompute the transformed sequences for x = 0, 1, ..., 2^M - 1
    # But this is not feasible for large M
    
    # Instead, for each pair (i,j), we compute the difference vector and simulate the transformation until it becomes zero
    
    # So, let's implement this
    
    # First, define a function to apply the transformation step
    def transform(v):
        new_v = [0] * M
        for k in range(M):
            new_v[k] = sum(v[:k+1]) % 2
        return new_v
    
    # Now, for each pair (i,j), compute the difference vector and simulate the transformation
    total = 0
    for i in range(N):
        for j in range(i, N):
            if i == j:
                total += 0
                continue
            v = [A[i][k] ^ A[j][k] for k in range(M)]
            x = 0
            while True:
                if all(v[k] == 0 for k in range(M)):
                    break
                v = transform(v)
                x += 1
                if x > 2**M:
                    x = 0
                    break
            total += x
    print(total % MOD)

if __name__ == "__main__":
    main()