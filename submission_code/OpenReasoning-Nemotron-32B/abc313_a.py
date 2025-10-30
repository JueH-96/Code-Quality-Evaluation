n = int(input().strip())
scores = list(map(int, input().split()))

if n == 1:
	print(0)
else:
	p1 = scores[0]
	max_rest = max(scores[1:])
	if p1 > max_rest:
		print(0)
	else:
		print(max_rest - p1 + 1)