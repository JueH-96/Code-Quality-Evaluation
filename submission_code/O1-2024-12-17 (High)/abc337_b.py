def main():
    import sys
    S = sys.stdin.readline().strip()

    i = 0
    n = len(S)

    # Skip A's
    while i < n and S[i] == 'A':
        i += 1
    # Skip B's
    while i < n and S[i] == 'B':
        i += 1
    # Skip C's
    while i < n and S[i] == 'C':
        i += 1

    if i == n:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()