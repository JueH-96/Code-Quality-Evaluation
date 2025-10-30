N, D = map(int, input().split())
S = list(input())

for _ in range(D):
    rightmost_cookie = -1
    for i in range(N):
        if S[i] == '@':
            rightmost_cookie = i
    S[rightmost_cookie] = '.'

print("".join(S))