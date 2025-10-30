def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    X = list(map(int, data[2:2+M]))
    
    total_sum = 0
    for k in range(M-1):
        d = abs(X[k] - X[k+1])
        min_d = min(d, N - d)
        total_sum += min_d
    print(total_sum)

if __name__ == '__main__':
    main()