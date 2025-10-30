s = list(map(int, input().split()))

monotonic = True
for i in range(7):
  if s[i] > s[i+1]:
    monotonic = False
    break

valid_range = True
for x in s:
  if x < 100 or x > 675:
    valid_range = False
    break

multiple_of_25 = True
for x in s:
  if x % 25 != 0:
    multiple_of_25 = False
    break

if monotonic and valid_range and multiple_of_25:
  print("Yes")
else:
  print("No")