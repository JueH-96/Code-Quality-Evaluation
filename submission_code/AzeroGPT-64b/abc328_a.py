n, x = map(int, input().split())
s = list(map(int, input().split()))
result = sum(score for score in s if score <= x)
print(result)