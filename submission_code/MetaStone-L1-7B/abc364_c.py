def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    Y = int(data[2])
    
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+2*N]))
    
    # Sort A and B in descending order
    A.sort(reverse=True)
    B.sort(reverse=True)
    
    # Compute prefix sums for A and B
    prefixA = [0] * (N + 1)
    prefixB = [0] * (N + 1)
    
    for i in range(1, N + 1):
        prefixA[i] = prefixA[i-1] + A[i-1]
        prefixB[i] = prefixB[i-1] + B[i-1]
    
    # Find the minimal k
    for k in range(1, N + 1):
        if prefixA[k] > X or prefixB[k] > Y:
            print(k)
            return
    
    # If no such k found, return N
    print(N)
    
if __name__ == "__main__":
    main()