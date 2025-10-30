import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	H = list(map(int, data[1:1+n]))
	
	max_so_far = 0
	ans = []
	for i in range(n):
		if H[i] > max_so_far:
			max_so_far = H[i]
		result = (i + 1) * max_so_far + 1
		ans.append(str(result))
	
	print(" ".join(ans))

if __name__ == "__main__":
	main()