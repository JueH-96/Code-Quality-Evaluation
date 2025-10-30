def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    Q = int(input[1])
    T = list(map(int, input[2:]))
    count = [0] * (N + 1)
    for t in T:
        count[t] += 1
    odd_count = 0
    for i in range(1, N + 1):
        if count[i] % 2 == 1:
            odd_count += 1
    print(N - odd_count)

if __name__ == "__main__":
    main()