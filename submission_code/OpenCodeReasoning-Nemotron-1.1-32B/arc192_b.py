n = int(input().strip())
a = list(map(int, input().split()))
total = sum(a) - n
if total % 2 == 1:
	print("Fennec")
else:
	print("Snuke")