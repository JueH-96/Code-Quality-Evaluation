n = int(input().strip())
A = list(map(int, input().split()))
W = list(map(int, input().split()))

boxes = [[] for _ in range(n+1)]

for i in range(n):
	box_index = A[i]
	boxes[box_index].append(W[i])

total_weight = sum(W)
max_sum = 0
for j in range(1, n+1):
	if boxes[j]:
		max_sum += max(boxes[j])

print(total_weight - max_sum)