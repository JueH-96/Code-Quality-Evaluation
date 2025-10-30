def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    Ks = list(map(int, data[2:2+Q]))
    
    possible_K = set()
    for x in range(N+1):
        for y in range(N+1):
            K = N * (x + y) - x * y
            if 0 <= K <= N*N:
                possible_K.add(K)
    
    for K in Ks:
        if K in possible_K:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()