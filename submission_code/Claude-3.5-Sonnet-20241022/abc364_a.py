N = int(input())
dishes = [input() for _ in range(N)]

possible = True
for i in range(N-1):
    if dishes[i] == "sweet" and dishes[i+1] == "sweet":
        if i+2 < N:
            possible = False
            break

print("Yes" if possible else "No")