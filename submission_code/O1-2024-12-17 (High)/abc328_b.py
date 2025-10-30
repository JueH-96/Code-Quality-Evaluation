def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    D = list(map(int, data[1:]))

    def is_repdigit(x):
        s = str(x)
        return all(ch == s[0] for ch in s)

    answer = 0
    for i in range(1, N + 1):
        if is_repdigit(i):
            for j in range(1, D[i - 1] + 1):
                if is_repdigit(j):
                    answer += 1

    print(answer)

# Call main function
if __name__ == "__main__":
    main()