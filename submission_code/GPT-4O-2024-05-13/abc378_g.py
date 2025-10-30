# YOUR CODE HERE
import sys
input = sys.stdin.read

def count_permutations(A, B, M):
    AB = A * B
    dp = [[[0] * (B + 1) for _ in range(A + 1)] for _ in range(AB)]
    dp[0][0][0] = 1
    
    for length in range(1, AB):
        for lis in range(A + 1):
            for lds in range(B + 1):
                if dp[length - 1][lis][lds] == 0:
                    continue
                for new_lis in range(lis, A + 1):
                    for new_lds in range(lds, B + 1):
                        if new_lis == lis + 1 or new_lds == lds + 1:
                            dp[length][new_lis][new_lds] += dp[length - 1][lis][lds]
                            dp[length][new_lis][new_lds] %= M
    
    result = dp[AB - 1][A][B]
    print(result)

if __name__ == "__main__":
    A, B, M = map(int, input().strip().split())
    count_permutations(A, B, M)