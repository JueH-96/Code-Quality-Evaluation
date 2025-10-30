def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1:]
    
    # Determine the maximum length M among the given strings.
    M = max(len(s) for s in S)
    
    # Initialize a list T of length M, each element is a list of N '*' characters.
    # T[j] will eventually form the j-th output line (0-indexed).
    T = [['*'] * N for _ in range(M)]
    
    # Fill T according to the specification.
    # S[i][j] should become T[j][N - i - 1].
    for i, s in enumerate(S):
        for j, ch in enumerate(s):
            T[j][N - i - 1] = ch
    
    # Convert each row in T into a string and remove trailing '*'.
    for j in range(M):
        line = ''.join(T[j]).rstrip('*')
        print(line)

# Call main() so the solution actually runs.
if __name__ == "__main__":
    main()