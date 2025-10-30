def main():
	n = int(input().strip())
	A = list(map(int, input().split()))
	
	next_node = [0] * (n + 1)
	front = None
	for i in range(n):
		if A[i] == -1:
			front = i + 1
		else:
			next_node[A[i]] = i + 1
			
	current = front
	result = []
	while current != 0:
		result.append(str(current))
		current = next_node[current]
		
	print(" ".join(result))

if __name__ == "__main__":
	main()