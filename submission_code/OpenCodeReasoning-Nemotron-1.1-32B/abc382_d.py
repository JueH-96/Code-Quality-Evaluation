import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	N = int(data[0])
	M = int(data[1])
	T = M - 10 * N + 9
	k = N

	if T < 0:
		print(0)
		return

	results_tuples = []

	def dfs(i, s, path):
		if i == k:
			results_tuples.append(tuple(path))
			return
		for x in range(0, T - s + 1):
			dfs(i + 1, s + x, path + [x])
	
	dfs(0, 0, [])
	
	sequences = []
	for tup in results_tuples:
		s_val = 0
		seq = []
		for i in range(len(tup)):
			s_val += tup[i]
			element = s_val + 10 * i + 1
			seq.append(str(element))
		sequences.append(seq)
	
	print(len(sequences))
	for seq in sequences:
		print(" ".join(seq))

if __name__ == '__main__':
	main()