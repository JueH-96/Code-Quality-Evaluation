def main():
    import sys

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    S = input_data[1:]

    # M is the maximum length of the given strings
    M = max(len(s) for s in S)

    # For each j in [0..M-1], build T_j
    for j in range(M):
        row_chars = []
        # Collect characters from S_(N-1) down to S_0
        for k in range(N):
            # Index in S for the string from bottom to top is N - k - 1
            string_index = N - k - 1
            if j < len(S[string_index]):
                row_chars.append(S[string_index][j])
            else:
                row_chars.append('*')
        # Convert to string and strip trailing '*'
        # We don't want T_j to end with '*', so strip all trailing '*'
        t_j = ''.join(row_chars).rstrip('*')
        print(t_j)

# Don't forget to call main()
if __name__ == "__main__":
    main()