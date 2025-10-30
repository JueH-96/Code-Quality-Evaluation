def calculate_f(x, y):
    """Calculate the length of the longest common prefix of two strings."""
    i = 0
    while i < len(x) and i < len(y) and x[i] == y[i]:
        i += 1
    return i

def main():
    """Read input from stdin and calculate the sum of f(S_i, S_j) for all pairs of strings."""
    N = int(input())
    S = [input() for _ in range(N)]

    total = 0
    for i in range(N):
        for j in range(i + 1, N):
            total += calculate_f(S[i], S[j])

    print(total)

if __name__ == "__main__":
    main()