def count_arithmetic_subsequences(N, A):
    MOD = 998244353
    result = [0] * N
    
    # For k = 1, every element is a valid subsequence
    result[0] = N
    
    # For k = 2, count pairs (i, j) such that A[j] - A[i] is constant
    count_pairs = 0
    for i in range(N):
        for j in range(i + 1, N):
            if (A[j] - A[i]) % 2 == 0:
                count_pairs += 1
    result[1] = count_pairs % MOD
    
    # For k >= 3, we need to find valid subsequences of length k
    for k in range(3, N + 1):
        count_k = 0
        for i in range(N):
            for j in range(i + 1, N):
                d = A[j] - A[i]
                # We need to find how many elements can be added to form an arithmetic sequence
                count = 0
                for m in range(N):
                    if m != i and m != j and (A[m] - A[i]) * (j - i) == d * (m - i):
                        count += 1
                # We can choose (k-2) elements from the count found
                if count >= (k - 2):
                    from math import comb
                    count_k += comb(count, k - 2)
        
        result[k - 1] = count_k % MOD
    
    return result

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

result = count_arithmetic_subsequences(N, A)
print(" ".join(map(str, result)))