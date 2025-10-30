S = input().strip()

count = 0
index = 0
n = len(S)

while index < n:
    if S[index] != '0':
        count += 1
        index += 1
    else:
        if index + 1 < n and S[index + 1] == '0':
            count += 1
            index += 2
        else:
            count += 1
            index += 1

print(count)