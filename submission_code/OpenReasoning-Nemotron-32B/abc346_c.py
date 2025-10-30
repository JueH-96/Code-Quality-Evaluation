import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	k = int(data[1])
	A = list(map(int, data[2:2+n]))
	
	total = k * (k + 1) // 2
	present_set = set()
	for num in A:
		if num <= k:
			present_set.add(num)
	sum_present = sum(present_set)
	result = total - sum_present
	print(result)

if __name__ == '__main__':
	main()