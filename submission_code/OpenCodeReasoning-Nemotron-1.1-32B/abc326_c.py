import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	a = list(map(int, data[2:2+n]))
	a.sort()
	left = 0
	max_count = 0
	for right in range(n):
		while a[right] - a[left] >= m:
			left += 1
		current_count = right - left + 1
		if current_count > max_count:
			max_count = current_count
	print(max_count)

if __name__ == "__main__":
	main()