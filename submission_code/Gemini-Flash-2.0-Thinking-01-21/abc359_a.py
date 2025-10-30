n = int(input())
takahashi_count = 0
for _ in range(n):
    s = input()
    if s == "Takahashi":
        takahashi_count += 1
print(takahashi_count)