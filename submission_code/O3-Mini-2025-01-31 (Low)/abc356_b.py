def main():
    import sys
    input = sys.stdin.readline
    # Read number of foods and number of nutrients
    N, M = map(int, input().split())
    # Read the daily nutrient goals for M nutrients
    goals = list(map(int, input().split()))
    
    # Prepare a list to hold the total intake for each nutrient.
    total_intake = [0] * M
    
    # Process each of the N foods
    for _ in range(N):
        nutrients = list(map(int, input().split()))
        # Add the nutrient intake from this food to the respective totals
        for j in range(M):
            total_intake[j] += nutrients[j]
    
    # Check each nutrient's total intake against its goal.
    for j in range(M):
        if total_intake[j] < goals[j]:
            print("No")
            return
    
    # If all goals are met
    print("Yes")

if __name__ == '__main__':
    main()