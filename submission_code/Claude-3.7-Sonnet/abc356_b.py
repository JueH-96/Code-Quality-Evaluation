def main():
    # Read number of foods and nutrients
    N, M = map(int, input().split())
    
    # Read the goal for each nutrient
    goals = list(map(int, input().split()))
    
    # Initialize total intake for each nutrient
    total_intake = [0] * M
    
    # Process each food
    for _ in range(N):
        nutrients = list(map(int, input().split()))
        for j in range(M):
            total_intake[j] += nutrients[j]
    
    # Check if all nutrient goals are met
    for j in range(M):
        if total_intake[j] < goals[j]:
            print("No")
            return
    
    print("Yes")

if __name__ == "__main__":
    main()