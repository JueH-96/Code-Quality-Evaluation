S = input().strip()
n = len(S)
count = 0

for i in range(1, n+1):
    if S[i-1] == 'A':
        for k in range(i+2, n+1):
            if S[k-1] == 'C':
                if (i + k) % 2 == 0:
                    j = (i + k) // 2
                    if S[j-1] == 'B':
                        count += 1

print(count)