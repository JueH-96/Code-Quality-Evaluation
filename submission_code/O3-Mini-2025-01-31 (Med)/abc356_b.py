def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # Iterator for the input values
    it = iter(input_data)
    # Read N and M
    N = int(next(it))
    M = int(next(it))
    
    # Read the minimum required amounts for each nutrient (A)
    required = [int(next(it)) for _ in range(M)]
    
    # Initialize a list to store total amounts of each nutrient consumed
    totals = [0] * M
    
    # Process each food item
    for _ in range(N):
        # For each food, add the nutrient values to the totals
        for j in range(M):
            totals[j] += int(next(it))
    
    # Check if for every nutrient the goal is met
    for j in range(M):
        if totals[j] < required[j]:
            print("No")
            return
    
    print("Yes")

if __name__ == "__main__":
    main()