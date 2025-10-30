n = int(input())
s = ""
for i in range(1,n+1):
  if i%3 == 0:
    s += "x"
  else:
    s += "o"
print(s)