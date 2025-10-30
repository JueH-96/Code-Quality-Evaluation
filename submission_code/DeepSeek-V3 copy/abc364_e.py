def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    Y = int(data[2])
    
    A = []
    B = []
    index = 3
    for _ in range(N):
        A.append(int(data[index]))
        B.append(int(data[index+1]))
        index += 2
    
    # Initialize DP table
    # dp[k][x][y] represents whether it's possible to select k dishes with total sweetness x and total saltiness y
    # Since N is small (up to 80), we can manage with a 3D DP table
    # Initialize with False
    dp = [[[False] * (Y+1) for _ in range(X+1)] for _ in range(N+1)]
    dp[0][0][0] = True
    
    max_dishes = 0
    
    for i in range(N):
        a = A[i]
        b = B[i]
        for k in range(N, -1, -1):
            for x in range(X, -1, -1):
                for y in range(Y, -1, -1):
                    if dp[k][x][y]:
                        if x + a <= X and y + b <= Y:
                            dp[k+1][x+a][y+b] = True
                            if k+1 > max_dishes:
                                max_dishes = k+1
    
    print(max_dishes)

if __name__ == "__main__":
    main()