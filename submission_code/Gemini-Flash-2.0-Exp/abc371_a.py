s_ab, s_ac, s_bc = input().split()

ages = {'A': 0, 'B': 0, 'C': 0}

if s_ab == '<':
  ages['A'] -= 1
  ages['B'] += 1
else:
  ages['A'] += 1
  ages['B'] -= 1

if s_ac == '<':
  ages['A'] -= 1
  ages['C'] += 1
else:
  ages['A'] += 1
  ages['C'] -= 1

if s_bc == '<':
  ages['B'] -= 1
  ages['C'] += 1
else:
  ages['B'] += 1
  ages['C'] -= 1

sorted_ages = sorted(ages.items(), key=lambda item: item[1], reverse=True)

print(sorted_ages[1][0])