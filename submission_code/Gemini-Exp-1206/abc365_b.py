n = int(input())
a = list(map(int, input().split()))

b = sorted(a, reverse=True)
second_largest = b[1]

for i in range(n):
    if a[i] == second_largest:
        print(i + 1)
        break