import sys

def main():
    n, x, y = map(int, sys.stdin.readline().split())
    dishes = [tuple(map(int, sys.stdin.readline().split())) for _ in range(n)]
    
    # Initialize DP table
    dp = [[-1] * (y + 1) for _ in range(x + 1)]
    dp[0][0] = 0
    max_dishes = 0
    
    for a, b in dishes:
        for i in range(x, -1, -1):
            for j in range(y, -1, -1):
                if dp[i][j] != -1:
                    new_i = i + a
                    new_j = j + b
                    new_count = dp[i][j] + 1
                    if new_count > max_dishes:
                        max_dishes = new_count
                    if new_i <= x and new_j <= y:
                        if dp[new_i][new_j] < new_count:
                            dp[new_i][new_j] = new_count
    print(max_dishes)

if __name__ == "__main__":
    main()