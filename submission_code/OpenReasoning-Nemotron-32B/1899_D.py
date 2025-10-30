import sys
from collections import Counter

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		n = int(data[index])
		index += 1
		arr = list(map(int, data[index:index+n]))
		index += n
		cnt = Counter(arr)
		total_pairs = 0
		for count_val in cnt.values():
			total_pairs += count_val * (count_val - 1) // 2
		if 1 in cnt and 2 in cnt:
			total_pairs += cnt[1] * cnt[2]
		results.append(str(total_pairs))
	
	print("
".join(results))

if __name__ == '__main__':
	main()