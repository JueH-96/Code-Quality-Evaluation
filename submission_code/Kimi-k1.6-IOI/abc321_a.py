n = input().strip()
valid = True
for i in range(len(n)-1):
    if n[i] <= n[i+1]:
        valid = False
        break
print("Yes" if valid else "No")