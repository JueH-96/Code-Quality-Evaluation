def prefix_sum_mod_2(arr):
    """Calculate the prefix sum modulo 2 of an array"""
    result = [0] * len(arr)
    prefix_sum = 0
    for i in range(len(arr)):
        prefix_sum = (prefix_sum + arr[i]) % 2
        result[i] = prefix_sum
    return result

def f(seq_i, seq_j):
    """
    Calculate the smallest non-negative integer x such that seq_i becomes seq_j 
    after applying the operation x times, or 0 if no such x exists.
    
    The operation has a period of at most 4, so we only need to check up to 3 operations.
    """
    if seq_i == seq_j:
        return 0
    
    # Apply operation once
    seq_after_1 = prefix_sum_mod_2(seq_i)
    if seq_after_1 == seq_j:
        return 1
    
    # Apply operation twice
    seq_after_2 = prefix_sum_mod_2(seq_after_1)
    if seq_after_2 == seq_j:
        return 2
    
    # Apply operation three times
    seq_after_3 = prefix_sum_mod_2(seq_after_2)
    if seq_after_3 == seq_j:
        return 3
    
    # If we reach here, seq_i can never become seq_j
    return 0

def main():
    N, M = map(int, input().split())
    sequences = []
    for _ in range(N):
        sequences.append(list(map(int, input().split())))
    
    # Calculate the sum of f(i, j) for all i ≤ j
    MOD = 998244353
    total_sum = 0
    for i in range(N):
        for j in range(i, N):
            total_sum = (total_sum + f(sequences[i], sequences[j])) % MOD
    
    print(total_sum)

if __name__ == "__main__":
    main()