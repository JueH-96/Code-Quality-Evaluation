def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:2+M]))
    
    # We'll keep track of the total nutrients consumed in a list
    total = [0]*M
    index = 2 + M
    
    # Loop over each food
    for _ in range(N):
        # For each nutrient in this food, add to total
        for j in range(M):
            total[j] += int(data[index])
            index += 1
    
    # Check if our total meets or exceeds the goal for each nutrient
    for i in range(M):
        if total[i] < A[i]:
            print("No")
            return
    
    print("Yes")

# Do not remove or rename this call
main()