n, m = map(int, input().split())
a = list(map(int, input().split()))
foods = [list(map(int, input().split())) for _ in range(n)]
sums = [sum(col) for col in zip(*foods)]
for sa, ta in zip(sums, a):
    if sa < ta:
        print("No")
        exit()
print("Yes")