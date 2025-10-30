# YOUR CODE HERE
import sys
input = sys.stdin.read

MOD = 998244353

def count_operations(N, A):
    initial_grid = [(i % 2) for i in range(1, N + 1)]
    
    # Find segments where initial_grid differs from A
    segments = []
    i = 0
    while i < N:
        if initial_grid[i] != A[i]:
            start = i
            while i < N and initial_grid[i] != A[i]:
                i += 1
            segments.append((start, i - 1))
        else:
            i += 1
    
    # Calculate the number of ways to transform each segment
    total_ways = 1
    for start, end in segments:
        length = end - start + 1
        if length >= 3:
            total_ways = (total_ways * (length - 1)) % MOD
    
    return total_ways

def main():
    data = input().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    result = count_operations(N, A)
    print(result)

if __name__ == "__main__":
    main()