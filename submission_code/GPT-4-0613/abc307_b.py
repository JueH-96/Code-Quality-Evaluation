def is_palindrome(s):
    return s == s[::-1]

N = int(input().strip())
S = [input().strip() for _ in range(N)]

for i in range(N):
    for j in range(i+1, N):
        if is_palindrome(S[i] + S[j]) or is_palindrome(S[j] + S[i]):
            print('Yes')
            exit(0)
print('No')