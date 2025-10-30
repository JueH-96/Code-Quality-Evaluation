def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1:]  # List of strings S_1..S_N

    # Find the maximum length of the strings
    M = max(len(s) for s in S)

    # Construct and print M strings T_1..T_M
    # T_j is formed by looking at the j-th character of each string S_N..S_1
    # (S_N is "top", S_1 is "bottom").
    for j in range(M):
        row_chars = []
        for i in range(N):
            # We want to take the j-th character of S[N-1-i] if it exists; otherwise '*'
            idx = N - 1 - i
            if j < len(S[idx]):
                row_chars.append(S[idx][j])
            else:
                row_chars.append('*')
        # Remove trailing '*' from row_chars
        while row_chars and row_chars[-1] == '*':
            row_chars.pop()
        print("".join(row_chars))

def main():
    solve()

if __name__ == "__main__":
    main()