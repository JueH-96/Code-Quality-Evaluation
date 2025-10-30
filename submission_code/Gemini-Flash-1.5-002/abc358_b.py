# YOUR CODE HERE
n, a = map(int, input().split())
t = list(map(int, input().split()))

finish_time = 0
for i in range(n):
  arrival_time = t[i]
  if arrival_time > finish_time:
    finish_time = arrival_time + a
  else:
    finish_time += a
  print(finish_time)