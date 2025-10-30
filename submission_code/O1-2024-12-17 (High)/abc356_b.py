def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:2+M]))
    
    # Initialize sums of nutrients
    nutrient_sums = [0]*M
    
    # Read nutrient intake from each food
    index = 2 + M
    for _ in range(N):
        for j in range(M):
            nutrient_sums[j] += int(data[index])
            index += 1
    
    # Check if goals are met for each nutrient
    for j in range(M):
        if nutrient_sums[j] < A[j]:
            print("No")
            return
    
    print("Yes")

# Call main
if __name__ == "__main__":
    main()