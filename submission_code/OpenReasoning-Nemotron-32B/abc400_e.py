import sys
from bisect import bisect_right

def main():
	max_k = 1000000
	distinct = [0] * (max_k + 1)
	
	for i in range(2, max_k + 1):
		if distinct[i] == 0:
			for j in range(i, max_k + 1, i):
				distinct[j] += 1
				
	L = []
	for k in range(2, max_k + 1):
		if distinct[k] == 2:
			sq = k * k
			L.append(sq)
			
	data = sys.stdin.read().split()
	if not data:
		return
	q = int(data[0])
	queries = list(map(int, data[1:1 + q]))
	
	results = []
	for A in queries:
		idx = bisect_right(L, A)
		if idx == 0:
			results.append("0")
		else:
			results.append(str(L[idx - 1]))
			
	print("
".join(results))

if __name__ == "__main__":
	main()