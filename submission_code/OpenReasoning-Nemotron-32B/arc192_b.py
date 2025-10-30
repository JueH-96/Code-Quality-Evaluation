n = int(input().strip())
a = list(map(int, input().split()))
s = 0
for x in a:
	s += x - 1

if s % 2 == 1:
	print("Fennec")
else:
	print("Snuke")