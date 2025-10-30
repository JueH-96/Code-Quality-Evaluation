# YOUR CODE HERE
n, k = map(int, input().split())
a = list(map(int, input().split()))

quotients = []
for x in a:
  if x % k == 0:
    quotients.append(x // k)

quotients.sort()
print(*quotients)