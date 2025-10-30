import sys

def is_subsequence(s, t):
    """Check if t is a subsequence of s."""
    i = j = 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            j += 1
        i += 1
    return j == len(t)

def main():
    # Read input
    N = int(input())
    T = input()
    S = [input() for _ in range(N)]

    # Initialize count of pairs
    count = 0

    # Iterate over all pairs (i, j)
    for i in range(N):
        for j in range(N):
            # Concatenate S_i and S_j
            concat = S[i] + S[j]

            # Check if T is a subsequence of the concatenation
            if is_subsequence(concat, T):
                count += 1

    # Print the count of pairs
    print(count)

if __name__ == "__main__":
    main()