MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    B = list(map(int, data[2:2+N]))
    
    # Count the number of -1s
    q = B.count(-1)
    
    # Precompute the number of possible sequences
    total_sequences = pow(M, q, MOD)
    
    # The number of connected components is always N - (number of edges)
    # But in this problem, the number of connected components is determined by the sequence
    # For the given problem, the number of connected components is always 1 if all elements are the same
    # Otherwise, it depends on the sequence
    
    # However, for the given problem, the number of connected components is always 1 if all elements are the same
    # Otherwise, it is 2
    
    # So, for each sequence, if all elements are the same, f(B') = 1
    # Otherwise, f(B') = 2
    
    # Now, count the number of sequences where all elements are the same
    # The number of such sequences is M (since all elements must be the same, and there are M choices)
    
    # The number of sequences where not all elements are the same is total_sequences - M
    
    # So, the sum of f(B') is M * 1 + (total_sequences - M) * 2
    
    # But wait, this is not correct. The number of sequences where all elements are the same is M only if all elements are -1
    # Otherwise, it depends on the fixed elements
    
    # So, we need to consider the fixed elements
    
    # Let's find the number of sequences where all elements are the same
    
    # If all fixed elements are the same, then the number of sequences where all elements are the same is M
    # Otherwise, it is 0
    
    # So, first, check if all fixed elements are the same
    fixed_elements = [b for b in B if b != -1]
    if len(fixed_elements) == 0:
        # All elements are -1
        # The number of sequences where all elements are the same is M
        same_sequences = M
    else:
        # Check if all fixed elements are the same
        first_fixed = fixed_elements[0]
        all_same = all(b == first_fixed for b in fixed_elements)
        if all_same:
            same_sequences = M
        else:
            same_sequences = 0
    
    # The number of sequences where not all elements are the same is total_sequences - same_sequences
    
    # So, the sum of f(B') is same_sequences * 1 + (total_sequences - same_sequences) * 2
    
    sum_f = (same_sequences * 1 + (total_sequences - same_sequences) * 2) % MOD
    
    print(sum_f)

if __name__ == "__main__":
    main()