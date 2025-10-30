import sys

def is_11_22_string(s):
    n = len(s)
    if n % 2 == 0:
        return False
    
    mid = n // 2
    if s[mid] != '/':
        return False
    
    for i in range(mid):
        if s[i] != '1':
            return False
    
    for i in range(mid + 1, n):
        if s[i] != '2':
            return False
    
    return True

def max_11_22_subsequence(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i, n):
            if is_11_22_string(s[i:j+1]):
                dp[i][j] = j - i + 1
    
    max_len = 0
    for i in range(n):
        for j in range(i, n):
            max_len = max(max_len, dp[i][j])
    
    return max_len

def solve():
    N, Q = map(int, input().split())
    S = input()
    
    for _ in range(Q):
        L, R = map(int, input().split())
        print(max_11_22_subsequence(S[L-1:R]))

if __name__ == "__main__":
    solve()