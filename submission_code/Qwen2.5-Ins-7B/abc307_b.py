# N = int(input())
# strings = [input().strip() for _ in range(N)]

N = int(input())
strings = []
for _ in range(N):
    strings.append(input().strip())

for i in range(N):
    for j in range(N):
        if i != j:
            concatenated = strings[i] + strings[j]
            if concatenated == concatenated[::-1]:
                print("Yes")
                exit()
print("No")