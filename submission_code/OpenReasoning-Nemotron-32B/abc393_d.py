import sys

def main():
	data = sys.stdin.read().splitlines()
	n = int(data[0].strip())
	s = data[1].strip()
	
	ones = [i for i, char in enumerate(s) if char == '1']
	k = len(ones)
	if k == 0:
		print(0)
		return
		
	q = [ones[i] - i for i in range(k)]
	q.sort()
	median = q[k // 2]
	
	total_swaps = 0
	for num in q:
		total_swaps += abs(num - median)
		
	print(total_swaps)

if __name__ == "__main__":
	main()