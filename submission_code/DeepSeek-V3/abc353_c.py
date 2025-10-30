# YOUR CODE HERE
import sys

def main():
    MOD = 10**8
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Precompute the sum of all A_i
    total_sum = sum(A)
    
    # Compute the sum of f(A_i, A_j) for all i < j
    # f(A_i, A_j) = (A_i + A_j) % MOD
    # Sum over all i < j: (A_i + A_j) % MOD = [ (sum_{i<j} A_i) + (sum_{i<j} A_j) ] % MOD
    # sum_{i<j} A_i = sum_{i=1}^{N-1} A_i * (N - i)
    # sum_{i<j} A_j = sum_{j=2}^{N} A_j * (j-1)
    
    # Compute sum_{i=1}^{N-1} A_i * (N - i)
    sum_Ai = 0
    for i in range(N-1):
        sum_Ai += A[i] * (N - i - 1)
    
    # Compute sum_{j=2}^{N} A_j * (j-1)
    sum_Aj = 0
    for j in range(1, N):
        sum_Aj += A[j] * j
    
    # Total sum of (A_i + A_j) for all i < j
    total = sum_Ai + sum_Aj
    
    # Now, compute the sum modulo MOD
    # Since (a + b) % MOD = (a % MOD + b % MOD) % MOD, but since we are summing all pairs, we can compute the total sum and then take modulo MOD
    # However, since the sum can be large, we need to handle it carefully
    
    # The total sum is sum_{i<j} (A_i + A_j) = (N-1) * total_sum - sum_{i=1}^N A_i * (N - i)
    # Wait, no. Let's re-calculate:
    # sum_{i<j} (A_i + A_j) = sum_{i<j} A_i + sum_{i<j} A_j
    # sum_{i<j} A_i = sum_{i=1}^{N-1} A_i * (N - i)
    # sum_{i<j} A_j = sum_{j=2}^{N} A_j * (j-1)
    # So total = sum_Ai + sum_Aj
    
    # Now, to compute the sum modulo MOD, we can compute the sum of (A_i + A_j) % MOD for all i < j
    # But since (A_i + A_j) % MOD = (A_i % MOD + A_j % MOD) % MOD, we can precompute A_i % MOD for all i
    A_mod = [a % MOD for a in A]
    
    # Now, sum_{i<j} (A_mod[i] + A_mod[j]) % MOD
    # To compute this efficiently, we can count the number of pairs (i,j) where A_mod[i] + A_mod[j] >= MOD
    # The total sum is sum_{i<j} (A_mod[i] + A_mod[j]) - count * MOD
    
    # First, sort A_mod
    A_mod_sorted = sorted(A_mod)
    
    # Initialize the total sum
    total_mod = 0
    count = 0
    
    # Use two pointers to count the number of pairs where A_mod[i] + A_mod[j] >= MOD
    # For each i, find the smallest j > i such that A_mod[i] + A_mod[j] >= MOD
    # The number of such j is N - j
    for i in range(N):
        low = i + 1
        high = N - 1
        res = N
        while low <= high:
            mid = (low + high) // 2
            if A_mod_sorted[i] + A_mod_sorted[mid] >= MOD:
                res = mid
                high = mid - 1
            else:
                low = mid + 1
        count += N - res
    
    # Now, compute the total sum of (A_mod[i] + A_mod[j]) for all i < j
    # sum_{i<j} (A_mod[i] + A_mod[j]) = sum_{i<j} A_mod[i] + sum_{i<j} A_mod[j]
    # sum_{i<j} A_mod[i] = sum_{i=1}^{N-1} A_mod[i] * (N - i)
    # sum_{i<j} A_mod[j] = sum_{j=2}^{N} A_mod[j] * (j-1)
    
    sum_Ai_mod = 0
    for i in range(N-1):
        sum_Ai_mod += A_mod_sorted[i] * (N - i - 1)
    
    sum_Aj_mod = 0
    for j in range(1, N):
        sum_Aj_mod += A_mod_sorted[j] * j
    
    total_mod_sum = sum_Ai_mod + sum_Aj_mod
    
    # The final sum is total_mod_sum - count * MOD
    result = total_mod_sum - count * MOD
    
    print(result)

if __name__ == "__main__":
    main()