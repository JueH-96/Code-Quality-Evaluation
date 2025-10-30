n = int(input())
dishes = []
for i in range(n):
    dishes.append(input().strip())

for i in range(n - 1):
    if dishes[i] == "sweet" and dishes[i + 1] == "sweet":
        if i + 1 == n - 1:
            print("Yes")
        else:
            print("No")
        break
else:
    print("Yes")