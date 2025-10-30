N, L = map(int, input().split())
A = list(map(int, input().split()))

count = 0
for score in A:
  if score >= L:
    count += 1

print(count)