import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	n, m = map(int, data[0].split())
	s = data[1].strip()
	t = data[2].strip()
	
	sorted_t = sorted(t, reverse=True)
	j = 0
	res = []
	for i in range(n):
		if j < m and sorted_t[j] > s[i]:
			res.append(sorted_t[j])
			j += 1
		else:
			res.append(s[i])
	print(''.join(res))

if __name__ == "__main__":
	main()