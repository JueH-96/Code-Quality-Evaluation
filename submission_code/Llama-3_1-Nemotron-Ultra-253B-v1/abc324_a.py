n = int(input())
a = list(map(int, input().split()))
print("Yes" if len(set(a)) == 1 else "No")