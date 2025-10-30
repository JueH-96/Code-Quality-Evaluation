import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	
	n = int(data[0])
	M = int(data[1])
	F = [M + 1] * (M + 2)
	
	index = 2
	for _ in range(n):
		L = int(data[index])
		R = int(data[index + 1])
		index += 2
		if R < F[L]:
			F[L] = R
	
	for l in range(M, 0, -1):
		if F[l] > F[l + 1]:
			F[l] = F[l + 1]
	
	total = 0
	for l in range(1, M + 1):
		high_bound = min(F[l] - 1, M)
		if high_bound >= l:
			total += (high_bound - l + 1)
	
	print(total)

if __name__ == "__main__":
	main()