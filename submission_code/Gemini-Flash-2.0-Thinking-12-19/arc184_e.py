def transform(a):
  m = len(a)
  next_a = [0] * m
  current_sum = 0
  for k in range(m):
    current_sum = (current_sum + a[k]) % 2
    next_a[k] = current_sum
  return tuple(next_a)

def are_equal(a, b):
  return tuple(a) == tuple(b)

def calculate_f(a_i, a_j, m):
  current_a_i = list(a_i)
  current_a_j = list(a_j)
  for x in range(m + 1):
    if are_equal(current_a_i, current_a_j):
      return x
    current_a_i = list(transform(current_a_i))
    current_a_j = list(transform(current_a_j))
  return 0

n, m = map(int, input().split())
a_sequences = []
for _ in range(n):
  a_sequences.append(list(map(int, input().split())))

total_sum = 0
mod = 998244353
for i in range(n):
  for j in range(i, n):
    f_val = calculate_f(a_sequences[i], a_sequences[j], m)
    total_sum = (total_sum + f_val) % mod

print(total_sum)