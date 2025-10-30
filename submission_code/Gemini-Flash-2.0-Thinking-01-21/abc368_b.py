import sys

def solve():
  n = int(sys.stdin.readline())
  a = list(map(int, sys.stdin.readline().split()))
  operation_count = 0
  while count_positive(a) > 1:
    operation_count += 1
    a.sort(reverse=True)
    a[0] -= 1
    a[1] -= 1
  print(operation_count)

def count_positive(arr):
  count = 0
  for x in arr:
    if x > 0:
      count += 1
  return count

if __name__ == '__main__':
  solve()