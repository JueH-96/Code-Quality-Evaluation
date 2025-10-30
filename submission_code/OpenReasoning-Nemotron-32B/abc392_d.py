import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	index = 1
	dice_freq = []
	sizes = []
	
	for i in range(n):
		k = int(data[index])
		index += 1
		sizes.append(k)
		faces = list(map(int, data[index:index+k]))
		index += k
		freq = {}
		for x in faces:
			freq[x] = freq.get(x, 0) + 1
		dice_freq.append(freq)
	
	mapping = {}
	for die_index, freq_dict in enumerate(dice_freq):
		for number, count in freq_dict.items():
			if number not in mapping:
				mapping[number] = []
			mapping[number].append((die_index, count))
	
	S = [[0] * n for _ in range(n)]
	
	for number, lst in mapping.items():
		m = len(lst)
		if m < 2:
			continue
		for i in range(m):
			die1, cnt1 = lst[i]
			for j in range(i+1, m):
				die2, cnt2 = lst[j]
				a = min(die1, die2)
				b = max(die1, die2)
				S[a][b] += cnt1 * cnt2
				
	max_prob = 0.0
	for i in range(n):
		for j in range(i+1, n):
			total_outcomes = sizes[i] * sizes[j]
			prob = S[i][j] / total_outcomes
			if prob > max_prob:
				max_prob = prob
				
	print("{:.15f}".format(max_prob))

if __name__ == "__main__":
	main()