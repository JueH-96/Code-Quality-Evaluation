n = int(input())
a = list(map(int, input().split()))

valid = True
for i in range(n-1):
    if a[i] >= a[i+1]:
        valid = False
        break

print("Yes" if valid else "No")