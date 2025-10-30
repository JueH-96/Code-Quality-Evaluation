n, a = map(int, input().split())
t = list(map(int, input().split()))

finish_times = []
current_time = 0

for i in range(n):
  if current_time <= t[i]:
    current_time = t[i] + a
    finish_times.append(current_time)
  else:
    current_time += a
    finish_times.append(current_time)

for time in finish_times:
  print(time)