import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	W = list(map(int, data[1+n:1+2*n]))
	
	total_weight = sum(W)
	boxes = [[] for _ in range(n)]
	
	for i in range(n):
		box_index = A[i] - 1
		boxes[box_index].append(W[i])
	
	max_sum = 0
	for j in range(n):
		if boxes[j]:
			max_sum += max(boxes[j])
	
	print(total_weight - max_sum)

if __name__ == "__main__":
	main()