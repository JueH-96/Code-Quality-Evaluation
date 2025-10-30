# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:N+2]))
    B = list(map(int, data[N+2:N+2+M]))
    
    # Sort A in ascending order
    A.sort()
    # Sort B in ascending order
    B.sort()
    
    # Check if it's possible to satisfy the condition
    if A[-M:] < B:
        print(-1)
        return
    
    # Use a greedy approach to find the minimum total amount of money
    total_cost = 0
    j = 0
    for b in B:
        while j < N and A[j] < b:
            j += 1
        if j == N:
            print(-1)
            return
        total_cost += A[j]
        j += 1
    
    print(total_cost)

if __name__ == "__main__":
    main()