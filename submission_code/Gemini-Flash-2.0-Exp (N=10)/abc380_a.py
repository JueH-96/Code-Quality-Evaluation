n = input()
counts = {}
for digit in n:
  if digit in counts:
    counts[digit] += 1
  else:
    counts[digit] = 1

if '1' not in counts or counts['1'] != 1:
  print("No")
elif '2' not in counts or counts['2'] != 2:
  print("No")
elif '3' not in counts or counts['3'] != 3:
  print("No")
else:
  
  other_digits = [digit for digit in counts if digit not in ['1','2','3']]
  if len(other_digits) > 0:
    print("No")
  else:
    print("Yes")