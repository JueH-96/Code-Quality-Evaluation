n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))
a_set = set(a)
c = sorted(a + b)
for i in range(len(c) - 1):
    if c[i] in a_set and c[i+1] in a_set:
        print("Yes")
        exit()
print("No")