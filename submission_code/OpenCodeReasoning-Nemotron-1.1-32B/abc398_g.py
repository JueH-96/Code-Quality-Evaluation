import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	total_edges = n * (n - 1) // 2 - m
	if total_edges % 2 == 1:
		print("Aoki")
	else:
		print("Takahashi")

if __name__ == "__main__":
	main()