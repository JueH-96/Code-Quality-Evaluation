# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = list(map(int, data[1:N+1]))
    A = list(map(int, data[N+1:2*N+1]))
    B = list(map(int, data[2*N+1:3*N+1]))
    
    max_total = 0
    
    # Iterate over possible number of dish A servings
    # Since the maximum possible is limited by the ingredient with the smallest Q_i / A_i
    max_a = float('inf')
    for i in range(N):
        if A[i] == 0:
            continue
        max_a = min(max_a, Q[i] // A[i])
    
    for a in range(max_a + 1):
        # Calculate remaining ingredients after making 'a' servings of dish A
        remaining = [Q[i] - a * A[i] for i in range(N)]
        
        # Determine the maximum number of dish B servings possible with the remaining ingredients
        max_b = float('inf')
        for i in range(N):
            if B[i] == 0:
                continue
            max_b = min(max_b, remaining[i] // B[i])
        
        total = a + max_b
        if total > max_total:
            max_total = total
    
    print(max_total)

if __name__ == "__main__":
    main()