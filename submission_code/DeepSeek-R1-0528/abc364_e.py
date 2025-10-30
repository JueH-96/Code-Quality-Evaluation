import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    X = int(data[1])
    Y = int(data[2])
    dishes = []
    index = 3
    for i in range(n):
        a = int(data[index])
        b = int(data[index+1])
        index += 2
        dishes.append((a, b))
    
    dp = [[-1] * (Y+1) for _ in range(X+1)]
    dp[0][0] = 0
    ans = 0
    for a, b in dishes:
        new_dp = [row[:] for row in dp]
        for x in range(X+1):
            for y in range(Y+1):
                if dp[x][y] == -1:
                    continue
                nx = x + a
                ny = y + b
                if nx > X:
                    nx = X
                if ny > Y:
                    ny = Y
                if new_dp[nx][ny] < dp[x][y] + 1:
                    new_dp[nx][ny] = dp[x][y] + 1
        dp = new_dp
    
    for x in range(X+1):
        for y in range(Y+1):
            if dp[x][y] > ans:
                ans = dp[x][y]
    print(ans)

if __name__ == '__main__':
    main()