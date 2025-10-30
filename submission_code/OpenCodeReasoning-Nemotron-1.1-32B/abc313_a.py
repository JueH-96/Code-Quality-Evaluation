n = int(input().strip())
scores = list(map(int, input().split()))

if n == 1:
	max_other = 0
else:
	max_other = max(scores[1:])

x = max(0, max_other - scores[0] + 1)
print(x)