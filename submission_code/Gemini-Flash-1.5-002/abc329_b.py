# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

max_val = max(a)
second_max = -1

for x in a:
  if x != max_val and x > second_max:
    second_max = x

print(second_max)