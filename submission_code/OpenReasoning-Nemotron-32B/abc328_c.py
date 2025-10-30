import sys

def main():
	data = sys.stdin.read().splitlines()
	n, q = map(int, data[0].split())
	s = data[1].strip()
	
	arr = [0] * n
	for i in range(n-1):
		if s[i] == s[i+1]:
			arr[i] = 1
	
	pref = [0] * (n+1)
	for i in range(1, n+1):
		pref[i] = pref[i-1] + arr[i-1]
	
	out_lines = []
	for i in range(2, 2 + q):
		l, r = map(int, data[i].split())
		ans = pref[r-1] - pref[l-1]
		out_lines.append(str(ans))
	
	sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
	main()