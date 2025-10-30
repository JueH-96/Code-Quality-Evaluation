def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    A = list(map(int, data[2:2+N-1]))
    
    S = sorted(A)
    min_an = -1
    
    for AN in range(0, 101):
        T = S + [AN]
        T.sort()
        sum_middle = sum(T[1:N-1])
        if sum_middle >= X:
            min_an = AN
            break
    
    print(min_an if min_an >= 0 else -1)

if __name__ == "__main__":
    main()