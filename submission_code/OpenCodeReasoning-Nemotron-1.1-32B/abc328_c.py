import sys

def main():
	data = sys.stdin.read().splitlines()
	n, q = map(int, data[0].split())
	s = data[1].strip()
	
	A = [0] * n
	for i in range(n - 1):
		if s[i] == s[i + 1]:
			A[i] = 1
			
	prefix = [0] * (n + 1)
	for i in range(1, n + 1):
		prefix[i] = prefix[i - 1] + A[i - 1]
		
	output_lines = []
	for i in range(2, 2 + q):
		l, r = map(int, data[i].split())
		res = prefix[r - 1] - prefix[l - 1]
		output_lines.append(str(res))
	
	sys.stdout.write("
".join(output_lines))

if __name__ == "__main__":
	main()