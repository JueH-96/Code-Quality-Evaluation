S = input()
ans = 'Yes'
for i in range(1,16,2):
    if S[i] != '0':
        ans = 'No'
        break
print(ans)