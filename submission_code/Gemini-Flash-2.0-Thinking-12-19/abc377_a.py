s = input()
count_a = s.count('A')
count_b = s.count('B')
count_c = s.count('C')

if count_a == 1 and count_b == 1 and count_c == 1:
  print("Yes")
else:
  print("No")