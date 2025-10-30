import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	H = list(map(int, data[1:1+n]))
	ans = [0] * n
	stack = []
	for i in range(n-1, -1, -1):
		while stack and H[i] > H[stack[-1]]:
			ans[i] += ans[stack.pop()] + 1
		stack.append(i)
	print(" ".join(map(str, ans)))

if __name__ == '__main__':
	main()