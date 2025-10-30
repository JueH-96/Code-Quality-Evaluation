def main():
	import sys
	from collections import defaultdict

	data = sys.stdin.read().split()
	n = int(data[0])
	arr = list(map(int, data[1:1+3*n]))
	
	pos_dict = defaultdict(list)
	for idx, num in enumerate(arr):
		pos_dict[num].append(idx)
	
	num_mid = []
	for num in range(1, n+1):
		positions = pos_dict[num]
		mid_index = positions[1]
		num_mid.append((num, mid_index))
	
	num_mid.sort(key=lambda x: x[1])
	
	result = [str(item[0]) for item in num_mid]
	print(" ".join(result))

if __name__ == "__main__":
	main()