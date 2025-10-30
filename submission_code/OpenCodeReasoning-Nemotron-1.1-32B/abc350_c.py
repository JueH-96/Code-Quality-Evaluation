import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	pos = [0] * (n + 1)
	for idx in range(n):
		num = A[idx]
		pos[num] = idx
		
	operations = []
	
	for i in range(n):
		if A[i] == i + 1:
			continue
			
		j = pos[i + 1]
		x = A[i]
		y = i + 1
		
		pos[x] = j
		pos[y] = i
		
		A[i], A[j] = A[j], A[i]
		
		operations.append((i + 1, j + 1))
		
	print(len(operations))
	for op in operations:
		print(op[0], op[1])

if __name__ == "__main__":
	main()