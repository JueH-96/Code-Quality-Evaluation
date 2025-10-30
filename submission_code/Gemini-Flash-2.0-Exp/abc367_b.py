x = input()
x = float(x)
x = str(x)
if '.' in x:
  x = x.rstrip('0')
  if x.endswith('.'):
    x = x[:-1]
print(x)