import itertools
import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	k = int(data[1])
	R = list(map(int, data[2:2+n]))
	ranges_list = [range(1, r + 1) for r in R]
	for seq in itertools.product(*ranges_list):
		total = sum(seq)
		if total % k == 0:
			print(" ".join(map(str, seq)))

if __name__ == "__main__":
	main()