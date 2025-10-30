n = int(input())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

result = max(a) + max(b)
print(result)