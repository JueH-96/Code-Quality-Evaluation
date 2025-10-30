N = int(input())
count = 0
for _ in range(N):
    name = input().strip()
    if name == "Takahashi":
        count += 1
print(count)