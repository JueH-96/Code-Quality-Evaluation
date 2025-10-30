# YOUR CODE HERE
n, m = map(int, input().split())
a = list(map(int, input().split()))

counts = {}
for x in a:
  counts[x] = counts.get(x, 0) + 1

for i in range(1, m + 1):
  current_counts = {}
  for j in range(i):
    current_counts[a[j]] = current_counts.get(a[j], 0) + 1
  
  max_votes = 0
  winner = -1
  for candidate, num_votes in current_counts.items():
    if num_votes > max_votes:
      max_votes = num_votes
      winner = candidate
    elif num_votes == max_votes and candidate < winner:
      winner = candidate
  print(winner)