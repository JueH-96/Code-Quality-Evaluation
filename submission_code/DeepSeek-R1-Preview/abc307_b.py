n = int(input())
strings = [input().strip() for _ in range(n)]

for i in range(n):
    for j in range(n):
        if i != j:
            combined = strings[i] + strings[j]
            if combined == combined[::-1]:
                print("Yes")
                exit()

print("No")