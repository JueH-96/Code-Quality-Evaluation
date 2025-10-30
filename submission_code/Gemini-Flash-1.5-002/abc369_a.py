# YOUR CODE HERE
a, b = map(int, input().split())

count = set()
count.add(2 * a - b)
count.add(2 * b - a)
count.add(a + b - a)

if a == b:
  print(1)
else:
  print(len(count))