# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and M
    N, M = map(int, data[0].split())
    
    # Read the goal amounts for each nutrient
    A = list(map(int, data[1].split()))
    
    # Initialize a list to accumulate the total intake for each nutrient
    total_intake = [0] * M
    
    # Read the food intake data
    for i in range(2, 2 + N):
        X = list(map(int, data[i].split()))
        for j in range(M):
            total_intake[j] += X[j]
    
    # Check if the total intake meets the goals
    for j in range(M):
        if total_intake[j] < A[j]:
            print("No")
            return
    
    print("Yes")

if __name__ == "__main__":
    main()