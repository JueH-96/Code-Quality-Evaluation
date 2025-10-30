def main():
    N, M = map(int, input().split())
    X = list(map(int, input().split()))
    X = [x-1 for x in X]
    X.sort()
    D = [0]*(M-1)
    for i in range(M-1):
        D[i] = X[i+1] - X[i]
    D.sort()
    print(sum(D[:M-1]))

if __name__ == "__main__":
    main()