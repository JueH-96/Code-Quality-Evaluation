def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    K_values = [int(data[i + 2]) for i in range(Q)]
    
    possible_K = set()
    
    for R in range(N + 1):
        for C in range(N + 1):
            if R > 0 and C > 0:
                # Possible only if:
                # - All R rows have sum N and all C columns have sum N
                # - Or, all R rows have sum 0 and all C columns have sum 0
                possible_K.add(R * N + C * N - R * C)
            else:
                # Other configurations are possible
                possible_K.add(R * N + C * N - R * C)
    
    for K in K_values:
        if K in possible_K:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()