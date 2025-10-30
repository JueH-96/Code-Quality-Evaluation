def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = list(map(int, data[1:N+1]))
    A = list(map(int, data[N+1:2*N+1]))
    B = list(map(int, data[2*N+1:3*N+1]))
    
    # We will use a brute force approach with a reasonable search space
    # Since N is small (max 10), we can afford to try combinations of servings
    
    # Maximum possible servings for each dish if only that dish is made
    max_a = min(Q[i] // A[i] if A[i] > 0 else float('inf') for i in range(N))
    max_b = min(Q[i] // B[i] if B[i] > 0 else float('inf') for i in range(N))
    
    max_servings = 0
    
    # Try all combinations of servings of A from 0 to max_a
    for servings_a in range(max_a + 1):
        # Calculate the maximum servings of B that can be made after servings_a of A are made
        max_possible_b = float('inf')
        for i in range(N):
            if B[i] > 0:
                remaining_ingredient = Q[i] - servings_a * A[i]
                max_possible_b = min(max_possible_b, remaining_ingredient // B[i])
        
        # Total servings is the sum of servings of A and the possible servings of B
        total_servings = servings_a + max_possible_b
        max_servings = max(max_servings, total_servings)
    
    print(max_servings)

if __name__ == "__main__":
    main()