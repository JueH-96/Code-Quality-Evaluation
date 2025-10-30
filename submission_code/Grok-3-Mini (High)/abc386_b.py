import sys
S = sys.stdin.readline().strip()
answer = 0
i = 0
length = len(S)
while i < length:
    if S[i] != '0':
        answer += 1
        i += 1
    else:
        len_run = 0
        while i < length and S[i] == '0':
            len_run += 1
            i += 1
        answer += (len_run + 1) // 2
print(answer)