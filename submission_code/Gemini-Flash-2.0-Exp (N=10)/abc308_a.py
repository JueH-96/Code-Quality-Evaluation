s = list(map(int, input().split()))

valid = True

# Check non-decreasing
for i in range(7):
  if s[i] > s[i+1]:
    valid = False
    break

if valid:
  # Check range and multiples of 25
  for x in s:
    if not (100 <= x <= 675 and x % 25 == 0):
      valid = False
      break

if valid:
  print("Yes")
else:
  print("No")