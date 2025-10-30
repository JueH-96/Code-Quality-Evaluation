# YOUR CODE HERE
import sys
input = sys.stdin.read
MOD = 998244353

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    A = list(map(int, data[3:]))
    
    # Precompute powers of 2 modulo MOD
    pow2 = [1] * (N + 1)
    for i in range(1, N + 1):
        pow2[i] = (pow2[i - 1] * 2) % MOD
    
    # Calculate the sum of scores
    total_sum = 0
    for i in range(1, 1 << N):
        subset = []
        for j in range(N):
            if i & (1 << j):
                subset.append(A[j])
        
        if len(subset) % M == 0:
            xor_sum = 0
            for num in subset:
                xor_sum ^= num
            total_sum = (total_sum + pow(xor_sum, K, MOD)) % MOD
    
    print(total_sum)

if __name__ == "__main__":
    main()