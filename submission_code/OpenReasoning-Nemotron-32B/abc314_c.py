import sys

def main():
	data = sys.stdin.read().splitlines()
	n, m = map(int, data[0].split())
	s = data[1].strip()
	colors = list(map(int, data[2].split()))
	
	groups = [[] for _ in range(m+1)]
	
	for i in range(n):
		c = colors[i]
		groups[c].append(i)
		
	res = [''] * n
	
	for color in range(1, m+1):
		indices = groups[color]
		k = len(indices)
		if k == 0:
			continue
		res[indices[0]] = s[indices[-1]]
		for j in range(k-1):
			res[indices[j+1]] = s[indices[j]]
			
	print(''.join(res))

if __name__ == '__main__':
	main()