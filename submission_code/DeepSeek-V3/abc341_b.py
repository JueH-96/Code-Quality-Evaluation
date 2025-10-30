def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    S = []
    T = []
    for i in range(N-1):
        S.append(int(data[N+1 + 2*i]))
        T.append(int(data[N+2 + 2*i]))
    
    # Iterate from the first country to the (N-1)th country
    for i in range(N-1):
        # Calculate the maximum number of operations possible
        operations = A[i] // S[i]
        # Update the current and next country's currency
        A[i] -= operations * S[i]
        A[i+1] += operations * T[i]
    
    print(A[-1])

if __name__ == "__main__":
    main()