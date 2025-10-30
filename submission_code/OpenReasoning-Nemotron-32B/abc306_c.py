n = int(input().strip())
arr = list(map(int, input().split()))
positions = [[] for _ in range(n+1)]

for idx, num in enumerate(arr):
	positions[num].append(idx + 1)

events = []
for num in range(1, n+1):
	mid = positions[num][1]
	events.append((mid, num))

events.sort(key=lambda x: x[0])

result = [str(num) for _, num in events]
print(" ".join(result))