import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	H = list(map(int, data[1:1+n]))
	
	prev_greater = [-1] * n
	stack = []
	for i in range(n):
		while stack and H[stack[-1]] <= H[i]:
			stack.pop()
		if stack:
			prev_greater[i] = stack[-1]
		else:
			prev_greater[i] = -1
		stack.append(i)
	
	diff = [0] * (n + 2)
	
	for j in range(n):
		if prev_greater[j] == -1:
			left_bound = 0
		else:
			left_bound = prev_greater[j]
		
		if left_bound <= j - 1:
			diff[left_bound] += 1
			diff[j] -= 1
			
	res = []
	cur = 0
	for i in range(n):
		cur += diff[i]
		res.append(str(cur))
		
	print(" ".join(res))

if __name__ == "__main__":
	main()