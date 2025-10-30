n, m = map(int, input().split())
A = list(map(int, input().split()))
totals = [0] * m

for _ in range(n):
	food_data = list(map(int, input().split()))
	for j in range(m):
		totals[j] += food_data[j]

if all(totals[j] >= A[j] for j in range(m)):
	print("Yes")
else:
	print("No")