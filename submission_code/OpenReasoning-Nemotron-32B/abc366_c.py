import sys

def main():
	n_max = 1000000
	q = int(sys.stdin.readline())
	freq = [0] * (n_max + 1)
	distinct_count = 0
	for _ in range(q):
		data = sys.stdin.readline().split()
		if data[0] == '1':
			x = int(data[1])
			if freq[x] == 0:
				distinct_count += 1
			freq[x] += 1
		elif data[0] == '2':
			x = int(data[1])
			freq[x] -= 1
			if freq[x] == 0:
				distinct_count -= 1
		else:  # data[0] == '3'
			print(distinct_count)

if __name__ == '__main__':
	main()