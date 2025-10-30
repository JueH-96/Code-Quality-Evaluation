def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    S = list(data[2])

    count = 0
    i = 0
    while i <= N - K:
        # Check if the substring of length K is all 'O'
        if all(S[j] == 'O' for j in range(i, i + K)):
            # Eat the strawberry (mark these K teeth as cavities)
            for j in range(i, i + K):
                S[j] = 'X'
            count += 1
            i += K  # Skip past these K teeth
        else:
            i += 1

    print(count)

if __name__ == "__main__":
    main()