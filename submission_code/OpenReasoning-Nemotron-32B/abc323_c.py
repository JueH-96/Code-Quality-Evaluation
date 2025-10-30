import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	
	first_line = data[0].split()
	N = int(first_line[0])
	M = int(first_line[1])
	
	second_line = data[1].split()
	A = list(map(int, second_line))
	
	S = []
	for i in range(2, 2+N):
		S.append(data[i].strip())
	
	total_scores = []
	for i in range(N):
		s = S[i]
		score_val = 0
		for j in range(M):
			if s[j] == 'o':
				score_val += A[j]
		total_scores.append(score_val + i+1)
	
	for i in range(N):
		others = []
		for j in range(N):
			if j != i:
				others.append(total_scores[j])
		max_other = max(others)
		
		if total_scores[i] > max_other:
			print(0)
		else:
			unsolved = []
			for j in range(M):
				if S[i][j] == 'x':
					unsolved.append(A[j])
			unsolved.sort(reverse=True)
			required = max_other - total_scores[i] + 1
			current_sum = 0
			k = 0
			for score in unsolved:
				current_sum += score
				k += 1
				if current_sum >= required:
					print(k)
					break

if __name__ == "__main__":
	main()