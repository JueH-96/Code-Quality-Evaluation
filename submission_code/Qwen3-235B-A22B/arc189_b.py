def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    X = list(map(int, input[1:N+1]))
    
    if N < 4:
        print(sum(X))
        return
    
    d = [X[i+1] - X[i] for i in range(N-1)]
    
    # Process odd and even indexed chains (0-based)
    # Odd chain: indices 0, 2, 4, ...
    chain_odd = []
    for i in range(0, N-1, 2):
        chain_odd.append(d[i])
    # Process forward
    for i in range(1, len(chain_odd)):
        if chain_odd[i] < chain_odd[i-1]:
            chain_odd[i] = chain_odd[i-1]
    # Update d
    idx = 0
    for i in range(0, N-1, 2):
        if idx < len(chain_odd):
            d[i] = chain_odd[idx]
            idx += 1
    
    # Even chain: indices 1, 3, 5, ...
    chain_even = []
    for i in range(1, N-1, 2):
        chain_even.append(d[i])
    # Process forward
    for i in range(1, len(chain_even)):
        if chain_even[i] < chain_even[i-1]:
            chain_even[i] = chain_even[i-1]
    # Update d
    idx = 0
    for i in range(1, N-1, 2):
        if idx < len(chain_even):
            d[i] = chain_even[idx]
            idx += 1
    
    # Reconstruct the sum
    total = X[0]
    current = X[0]
    for i in range(N-1):
        current += d[i]
        total += current
    
    print(total)

if __name__ == '__main__':
    main()