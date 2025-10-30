def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx +=1
    X = int(input[idx])
    idx +=1
    Y = int(input[idx])
    idx +=1
    A = list(map(int, input[idx:idx+N]))
    idx +=N
    B = list(map(int, input[idx:idx+N]))
    
    # Sort A and B in descending order
    A_sorted = sorted(A, reverse=True)
    B_sorted = sorted(B, reverse=True)
    
    # Compute prefix sums for A
    prefixA = [0] * (N + 1)
    for i in range(N):
        prefixA[i+1] = prefixA[i] + A_sorted[i]
    
    # Compute prefix sums for B
    prefixB = [0] * (N + 1)
    for i in range(N):
        prefixB[i+1] = prefixB[i] + B_sorted[i]
    
    # Find the minimal k
    for k in range(1, N+1):
        if prefixA[k] > X or prefixB[k] > Y:
            print(k)
            return
    print(N)

if __name__ == '__main__':
    main()