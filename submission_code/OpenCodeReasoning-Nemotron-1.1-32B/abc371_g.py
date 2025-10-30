import sys

def least_rotation(arr):
	n = len(arr)
	if n == 0:
		return 0
	s = arr + arr
	f = [-1] * (2 * n)
	k = 0
	for j in range(1, 2 * n):
		i = f[j - k - 1] if j > k else -1
		while i != -1 and s[j] != s[k + i + 1]:
			if s[j] < s[k + i + 1]:
				k = j - i - 1
			i = f[i]
		if i == -1 and s[j] != s[k]:
			if s[j] < s[k]:
				k = j
			f[j - k] = -1
		else:
			f[j - k] = i + 1
	return k

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	P = list(map(int, data[1:1+n]))
	A = list(map(int, data[1+n:1+2*n]))
	
	P0 = [x - 1 for x in P]
	vis = [False] * n
	res = [0] * n
	
	for i in range(n):
		if not vis[i]:
			cycle = []
			cur = i
			while not vis[cur]:
				vis[cur] = True
				cycle.append(cur)
				cur = P0[cur]
			vals = [A[x] for x in cycle]
			L = len(vals)
			if L == 1:
				s = 0
			else:
				s = least_rotation(vals)
			for idx in range(L):
				pos = cycle[idx]
				value = vals[(s + idx) % L]
				res[pos] = value
				
	print(" ".join(map(str, res)))

if __name__ == "__main__":
	main()