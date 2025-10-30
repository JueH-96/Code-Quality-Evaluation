# YOUR CODE HERE
def solve():
  n, k = map(int, input().split())
  s = input()
  
  black_indices = [i for i, char in enumerate(s) if char == 'B']
  
  if not black_indices:
    print(0)
    return
  
  count = 0
  i = 0
  while i < len(black_indices):
    count += 1
    j = i
    while j < len(black_indices) and black_indices[j] <= black_indices[i] + k -1:
      j += 1
    i = j
  print(count)


t = int(input())
for _ in range(t):
  solve()