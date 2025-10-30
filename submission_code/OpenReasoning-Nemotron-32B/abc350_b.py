import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	q = int(data[1])
	treatments = list(map(int, data[2:2+q]))
	
	freq = [0] * (n + 1)
	
	for tooth in treatments:
		freq[tooth] += 1
			
	count = 0
	for i in range(1, n + 1):
		if freq[i] % 2 == 0:
			count += 1
			
	print(count)

if __name__ == "__main__":
	main()