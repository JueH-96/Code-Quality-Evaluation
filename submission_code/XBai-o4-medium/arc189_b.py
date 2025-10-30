def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    X = list(map(int, input[1:N+1]))
    for i in range(N - 3):
        a = X[i]
        b = X[i+1]
        c = X[i+2]
        d = X[i+3]
        if a + d < b + c:
            new_b = a + d - c
            new_c = a + d - b
            X[i+1] = new_b
            X[i+2] = new_c
    print(sum(X))

if __name__ == "__main__":
    main()