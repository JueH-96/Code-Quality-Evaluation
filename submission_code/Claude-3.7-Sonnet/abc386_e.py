import itertools

def main():
    # Read input
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Calculate maximum XOR
    max_xor = 0
    for combination in itertools.combinations(range(N), K):
        # Calculate XOR of current combination
        current_xor = 0
        for idx in combination:
            current_xor ^= A[idx]
        
        # Update maximum
        max_xor = max(max_xor, current_xor)
    
    # Output result
    print(max_xor)

if __name__ == "__main__":
    main()