import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	
	n, d = map(int, data[0].split())
	snakes = []
	for i in range(1, n + 1):
		t, l = map(int, data[i].split())
		snakes.append((t, l))
	
	base_weights = [t * l for t, l in snakes]
	T_vals = [t for t, l in snakes]
	
	for k in range(1, d + 1):
		max_weight = max(base_weights[i] + T_vals[i] * k for i in range(n))
		print(max_weight)

if __name__ == '__main__':
	main()