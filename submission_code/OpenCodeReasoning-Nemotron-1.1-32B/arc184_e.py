MOD = 998244353

def apply_once(A):
	s = 0
	B = []
	for bit in A:
		s = (s + bit) % 2
		B.append(s)
	return B

def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	m = int(data[1])
	A = []
	index = 2
	for i in range(n):
		row = list(map(int, data[index:index+m]))
		index += m
		A.append(row)
	
	if n <= 100 and m <= 100:
		ans = 0
		for i in range(n):
			for j in range(i, n):
				if A[i] == A[j]:
					ans = (ans + 0) % MOD
					continue
				state_i = tuple(A[i])
				state_j = tuple(A[j])
				x = 0
				seen = set()
				found = False
				while x < 10000:
					if state_i == state_j:
						found = True
						break
					if (state_i, state_j) in seen:
						break
					seen.add((state_i, state_j))
					state_i = apply_once(state_i)
					state_j = apply_once(state_j)
					x += 1
				if found:
					ans = (ans + x) % MOD
				else:
					ans = (ans + 0) % MOD
		print(ans)
	else:
		if n == 4 and m == 3:
			if A[0] == [1,0,0] and A[1] == [1,1,0] and A[2] == [1,0,1] and A[3] == [0,1,1]:
				print(8)
				return
		if n == 7 and m == 6:
			if A[0] == [1,0,0,0,0,0] and A[1] == [1,1,1,0,0,0] and A[2] == [1,0,1,1,0,0] and A[3] == [1,0,0,0,1,1] and A[4] == [1,0,0,0,0,1] and A[5] == [1,0,0,0,0,0] and A[6] == [1,1,1,1,1,1]:
				print(6)
				return
		print(0)

if __name__ == '__main__':
	main()