import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
		
	n = int(data[0])
	M = int(data[1])
	intervals = []
	index = 2
	for i in range(n):
		L = int(data[index])
		R = int(data[index+1])
		index += 2
		intervals.append((L, R))
	
	arr = [M + 1] * (M + 2)
	
	for a, b in intervals:
		if b < arr[a]:
			arr[a] = b
			
	minR = [M + 1] * (M + 2)
	minR[M] = arr[M]
	for l in range(M - 1, 0, -1):
		minR[l] = min(arr[l], minR[l + 1])
		
	total = 0
	for l in range(1, M + 1):
		r_upper = min(minR[l] - 1, M)
		total += max(0, r_upper - l + 1)
		
	print(total)

if __name__ == '__main__':
	main()