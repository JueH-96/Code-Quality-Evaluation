def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+2*N]))
    
    def min_operations_to_match(x, y):
        # Calculate minimum operations to change x to y
        if x <= y:
            return min(y - x, x + M - y)
        else:
            return min(x - y, y + M - x)
    
    total_operations = 0
    
    for i in range(N):
        if i > 0:
            # Check if the operation is possible
            if A[i-1] == A[i]:
                print(-1)
                return
            if B[i-1] == B[i]:
                print(-1)
                return
        
        # Calculate the minimum operations needed for A[i] to become B[i]
        total_operations += min_operations_to_match(A[i], B[i])
    
    print(total_operations)

if __name__ == "__main__":
    main()