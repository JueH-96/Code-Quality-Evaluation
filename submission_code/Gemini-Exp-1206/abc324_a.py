n = int(input())
a = list(map(int, input().split()))

if all(x == a[0] for x in a):
    print("Yes")
else:
    print("No")