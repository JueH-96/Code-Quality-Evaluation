import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	H = list(map(int, data[1:1+n]))
	
	ans = [0] * n
	stack = [0]
	total = 0
	for i in range(n):
		while len(stack) > 1 and stack[-1] > H[i]:
			total -= stack.pop()
		if stack[-1] < H[i]:
			stack.append(H[i])
			total += H[i]
		ans[i] = total + i + 1
	
	print(" ".join(map(str, ans)))

if __name__ == '__main__':
	main()