n = int(input())
strings = [input().strip() for _ in range(n)]

found = False
for i in range(n):
    for j in range(n):
        if i != j:
            concat = strings[i] + strings[j]
            if concat == concat[::-1]:
                found = True
                break
    if found:
        break

print("Yes" if found else "No")