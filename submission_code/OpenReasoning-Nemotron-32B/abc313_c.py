import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	total_sum = sum(A)
	base = total_sum // n
	r = total_sum % n
	
	A.sort()
	total_abs = 0
	for i in range(n):
		if i < n - r:
			total_abs += abs(A[i] - base)
		else:
			total_abs += abs(A[i] - (base + 1))
			
	print(total_abs // 2)

if __name__ == "__main__":
	main()