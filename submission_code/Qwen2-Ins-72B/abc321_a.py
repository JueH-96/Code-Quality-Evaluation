N = input()
ans = 'Yes'
for i in range(len(N)-1):
    if N[i] <= N[i+1]:
        ans = 'No'
        break
print(ans)