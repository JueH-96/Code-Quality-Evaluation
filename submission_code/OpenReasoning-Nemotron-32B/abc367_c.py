import itertools
import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	k = int(data[1])
	R = list(map(int, data[2:2+n]))
	
	for seq in itertools.product(*(range(1, r + 1) for r in R)):
		if sum(seq) % k == 0:
			print(' '.join(map(str, seq)))

if __name__ == "__main__":
	main()