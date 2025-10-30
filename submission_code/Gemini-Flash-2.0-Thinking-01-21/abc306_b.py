input_line = input()
a_str = input_line.split()
a = [int(x) for x in a_str]
result = 0
power_of_2 = 1
for i in range(64):
  term = a[i] * power_of_2
  result += term
  power_of_2 *= 2
print(result)