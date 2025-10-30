import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	last_occurrence = [-1] * (n + 1)
	total = 0
	cur = 0
	for i in range(n):
		x = A[i]
		prev_index = last_occurrence[x]
		cur += i - prev_index
		total += cur
		last_occurrence[x] = i
	
	print(total)

if __name__ == "__main__":
	main()