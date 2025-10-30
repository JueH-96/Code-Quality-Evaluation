import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	stack = []
	for a in A:
		stack.append(a)
		while len(stack) >= 2 and stack[-1] == stack[-2]:
			x = stack.pop()
			stack.pop()
			stack.append(x + 1)
			
	print(len(stack))

if __name__ == "__main__":
	main()