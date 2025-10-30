import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	index = 1
	dice = []
	
	for i in range(n):
		k = int(data[index])
		index += 1
		arr = list(map(int, data[index:index+k]))
		index += k
		freq = {}
		for num in arr:
			freq[num] = freq.get(num, 0) + 1
		dice.append((k, freq))
	
	best = 0.0
	for i in range(n):
		k_i, dict_i = dice[i]
		for j in range(i+1, n):
			k_j, dict_j = dice[j]
			total_common = 0
			if len(dict_i) < len(dict_j):
				for num, cnt_i in dict_i.items():
					if num in dict_j:
						cnt_j = dict_j[num]
						total_common += cnt_i * cnt_j
			else:
				for num, cnt_j in dict_j.items():
					if num in dict_i:
						cnt_i = dict_i[num]
						total_common += cnt_i * cnt_j
			prob = total_common / (k_i * k_j)
			if prob > best:
				best = prob
				
	print("{:.15f}".format(best))

if __name__ == "__main__":
	main()