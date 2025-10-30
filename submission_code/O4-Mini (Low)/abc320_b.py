def main():
    S = input().strip()
    n = len(S)

    # Try all possible lengths from n down to 1
    for length in range(n, 0, -1):
        for start in range(0, n - length + 1):
            substr = S[start:start+length]
            if substr == substr[::-1]:
                print(length)
                return

if __name__ == "__main__":
    main()