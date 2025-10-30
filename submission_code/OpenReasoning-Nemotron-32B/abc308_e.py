import bisect

def main():
	n = int(input().strip())
	A = list(map(int, input().split()))
	s = input().strip()
	
	M = [[] for _ in range(3)]
	E = [[] for _ in range(3)]
	X = [[] for _ in range(3)]
	
	for i in range(n):
		if s[i] == 'M':
			M[A[i]].append(i)
		elif s[i] == 'E':
			E[A[i]].append(i)
		elif s[i] == 'X':
			X[A[i]].append(i)
			
	mex_table = {}
	for x in range(3):
		for y in range(3):
			for z in range(3):
				se = {x, y, z}
				m = 0
				while m in se:
					m += 1
				mex_table[(x, y, z)] = m
				
	ans = 0
	for x in range(3):
		for y in range(3):
			for z in range(3):
				list_M = M[x]
				list_E = E[y]
				list_X = X[z]
				if not list_E or not list_M or not list_X:
					continue
				total_count = 0
				for j in list_E:
					left_count = bisect.bisect_left(list_M, j)
					right_count = len(list_X) - bisect.bisect_right(list_X, j)
					total_count += left_count * right_count
				ans += mex_table[(x, y, z)] * total_count
				
	print(ans)

if __name__ == "__main__":
	main()