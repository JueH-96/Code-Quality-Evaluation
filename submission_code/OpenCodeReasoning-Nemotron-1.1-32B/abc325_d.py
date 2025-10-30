import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	products = []
	idx = 1
	for i in range(n):
		t = int(data[idx])
		d = int(data[idx + 1])
		idx += 2
		products.append((t, d))
	
	products.sort(key=lambda x: x[0])
	
	current_time = -10**18 - 1
	count = 0
	for t, d in products:
		start = max(current_time + 1, t)
		if start <= t + d:
			count += 1
			current_time = start
	
	print(count)

if __name__ == '__main__':
	main()