def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    X = list(map(int, data[2:2+M]))
    A = list(map(int, data[2+M:2+2*M]))
    
    # Calculate total stones
    total_stones = sum(A)
    
    # If total stones are not equal to N, it's impossible
    if total_stones != N:
        print(-1)
        return
    
    # Initialize stones in each cell
    stones = [0] * N
    for i in range(M):
        stones[X[i] - 1] = A[i]
    
    # Calculate the minimum number of operations
    operations = 0
    for i in range(N - 1):
        if stones[i] > 1:
            excess = stones[i] - 1
            stones[i] -= excess
            stones[i + 1] += excess
            operations += excess
    
    print(operations)