def main():
	n = int(input().strip())
	A = list(map(int, input().split()))
	stack = []
	for a in A:
		while stack and stack[-1] == a:
			stack.pop()
			a += 1
		stack.append(a)
	print(len(stack))

if __name__ == '__main__':
	main()