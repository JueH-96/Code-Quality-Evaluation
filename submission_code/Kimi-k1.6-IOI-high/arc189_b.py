def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    X = list(map(int, data[1:N+1]))
    
    for i in range(N-4, -1, -1):
        if X[i] + X[i+3] < X[i+1] + X[i+2]:
            new_i1 = X[i] + X[i+3] - X[i+2]
            new_i2 = X[i] + X[i+3] - X[i+1]
            X[i+1] = new_i1
            X[i+2] = new_i2
    
    print(sum(X))

if __name__ == "__main__":
    main()