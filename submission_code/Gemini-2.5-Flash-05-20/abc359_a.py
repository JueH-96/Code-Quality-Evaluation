N = int(input())
count_takahashi = 0
for _ in range(N):
    s = input()
    if s == "Takahashi":
        count_takahashi += 1
print(count_takahashi)