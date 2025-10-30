n = int(input())
strings = [input().strip() for _ in range(n)]

found = False
for i in range(n):
    for j in range(n):
        if i != j:
            concatenated = strings[i] + strings[j]
            if concatenated == concatenated[::-1]:
                found = True
                break
    if found:
        break

print("Yes" if found else "No")