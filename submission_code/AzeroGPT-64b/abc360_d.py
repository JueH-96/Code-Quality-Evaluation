def main():
    from sys import stdin, stdout
    input = stdin.readline
    N, T = map(int, input().split())
    S = input()
    X = list(map(int, input().split()))
    X0 = []
    X1 = []
    for i in range(N):
        if S[i] == '0':
            X0.append((T + 0.1 - X[i], i))
        else:
            X1.append((T + 0.1 + X[i], i))
    X0.sort()
    X1.sort()
    cnt = 0
    j = 0
    for _, i in X0:
        while j < len(X1) and X1[j][0] <= X0[n - 1 - j][0]:
            j += 1
        cnt += j
    stdout.write(str(cnt))

if __name__ == "__main__":
    main()