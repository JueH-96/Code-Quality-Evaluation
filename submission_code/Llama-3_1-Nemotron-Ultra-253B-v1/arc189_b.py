def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    X = list(map(int, input[1:N+1]))
    sum_total = sum(X)
    for i in range(N-3):
        a = X[i+1]
        b = X[i+2]
        x_i = X[i]
        x_i3 = X[i+3]
        if x_i + x_i3 < a + b:
            delta = (a + b) - (x_i + x_i3)
            sum_total -= 2 * delta
            new_a = x_i + x_i3 - a
            new_b = x_i + x_i3 - b
            X[i+1] = new_b
            X[i+2] = new_a
    print(sum_total)

if __name__ == "__main__":
    main()