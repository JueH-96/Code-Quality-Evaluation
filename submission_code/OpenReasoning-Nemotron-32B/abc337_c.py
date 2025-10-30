import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	next_node = [0] * (n + 1)
	front = None
	
	for i in range(1, n + 1):
		a_val = A[i - 1]
		if a_val == -1:
			front = i
		else:
			next_node[a_val] = i
			
	current = front
	result = []
	while current != 0:
		result.append(str(current))
		current = next_node[current]
		
	print(" ".join(result))

if __name__ == "__main__":
	main()