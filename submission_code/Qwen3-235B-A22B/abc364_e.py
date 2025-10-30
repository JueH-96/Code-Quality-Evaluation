import sys

def main():
    n, x, y = map(int, sys.stdin.readline().split())
    dishes = [tuple(map(int, sys.stdin.readline().split())) for _ in range(n)]
    
    # dp[k] is a dictionary: key is sumA, value is minimal sumB
    dp = [{} for _ in range(n+1)]
    dp[0][0] = 0
    
    for ai, bi in dishes:
        # Iterate k in reverse order to avoid reusing the same dish
        for k in range(n-1, -1, -1):
            if not dp[k]:
                continue
            current_states = list(dp[k].items())
            for a, b in current_states:
                new_a = a + ai
                new_b = b + bi
                if new_a <= x and new_b <= y:
                    # Update dp[k+1] with minimal B for given A
                    if new_a in dp[k+1]:
                        if new_b < dp[k+1][new_a]:
                            dp[k+1][new_a] = new_b
                    else:
                        dp[k+1][new_a] = new_b
    
    # Find the maximum k with non-empty dp[k]
    max_live = 0
    for k in range(n, -1, -1):
        if dp[k]:
            max_live = k
            break
    
    if max_live == n:
        print(max_live)
    else:
        print(max_live + 1)

if __name__ == "__main__":
    main()