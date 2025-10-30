# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read the number of contests and initial rating
    N, R = map(int, data[0].split())
    
    # Process each contest
    for i in range(1, N + 1):
        D_i, A_i = map(int, data[i].split())
        
        # Determine if Takahashi is subject to rating updates
        if D_i == 1:
            # Div. 1: rating update if 1600 <= R <= 2799
            if 1600 <= R <= 2799:
                R += A_i
        elif D_i == 2:
            # Div. 2: rating update if 1200 <= R <= 2399
            if 1200 <= R <= 2399:
                R += A_i
    
    # Output the final rating
    print(R)

if __name__ == "__main__":
    main()