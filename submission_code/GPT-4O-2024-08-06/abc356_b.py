# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    # Read N and M
    N = int(data[0])
    M = int(data[1])
    
    # Read A_i values
    A = list(map(int, data[2:2+M]))
    
    # Initialize total intake for each nutrient to zero
    total_intake = [0] * M
    
    # Read X_{i,j} values and accumulate the intake for each nutrient
    index = 2 + M
    for _ in range(N):
        for j in range(M):
            total_intake[j] += int(data[index])
            index += 1
    
    # Check if the goal is met for all nutrients
    for j in range(M):
        if total_intake[j] < A[j]:
            print("No")
            return
    
    print("Yes")

if __name__ == "__main__":
    main()