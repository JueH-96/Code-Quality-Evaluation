import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	B = list(map(int, data[1+n:1+n+n-1]))
	
	A_desc = sorted(A, reverse=True)
	B_desc = sorted(B, reverse=True)
	
	prefix_ok = [False] * n
	prefix_ok[0] = True
	for i in range(1, n):
		if prefix_ok[i-1] and (B_desc[i-1] >= A_desc[i-1]):
			prefix_ok[i] = True
		else:
			prefix_ok[i] = False
			
	suffix_ok = [False] * n
	if n >= 1:
		suffix_ok[n-1] = True
	for i in range(n-2, -1, -1):
		if B_desc[i] >= A_desc[i+1] and suffix_ok[i+1]:
			suffix_ok[i] = True
		else:
			suffix_ok[i] = False
			
	candidates = []
	for i in range(n):
		if prefix_ok[i] and suffix_ok[i]:
			candidates.append(A_desc[i])
			
	if candidates:
		print(min(candidates))
	else:
		print(-1)

if __name__ == "__main__":
	main()