import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	
	n, m = map(int, data[0].split())
	A = list(map(int, data[1].split()))
	S = []
	for i in range(2, 2 + n):
		S.append(data[i].strip())
	
	base_scores = []
	for i in range(n):
		s = 0
		for j in range(m):
			if S[i][j] == 'o':
				s += A[j]
		base_scores.append(s + (i + 1))
	
	results = []
	for i in range(n):
		other_max = max(base_scores[j] for j in range(n) if j != i)
		if base_scores[i] > other_max:
			results.append(0)
		else:
			unsolved = []
			for j in range(m):
				if S[i][j] == 'x':
					unsolved.append(A[j])
			unsolved.sort(reverse=True)
			total = base_scores[i]
			count = 0
			for score in unsolved:
				count += 1
				total += score
				if total > other_max:
					break
			results.append(count)
	
	for res in results:
		print(res)

if __name__ == "__main__":
	main()