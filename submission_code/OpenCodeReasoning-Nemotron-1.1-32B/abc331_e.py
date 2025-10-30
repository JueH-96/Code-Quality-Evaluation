import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	
	it = iter(data)
	n = int(next(it))
	m = int(next(it))
	l_val = int(next(it))
	
	a = [int(next(it)) for _ in range(n)]
	b = [int(next(it)) for _ in range(m)]
	
	forbidden_set = set()
	for _ in range(l_val):
		c = int(next(it))
		d = int(next(it))
		c0 = c - 1
		d0 = d - 1
		forbidden_set.add((c0, d0))
	
	top_main = list(enumerate(a))
	top_main.sort(key=lambda x: x[1], reverse=True)
	top_main = top_main[:l_val + 1]
	
	top_side = list(enumerate(b))
	top_side.sort(key=lambda x: x[1], reverse=True)
	top_side = top_side[:l_val + 1]
	
	ans = -10**18
	for main_index, a_val in top_main:
		for side_index, b_val in top_side:
			if (main_index, side_index) in forbidden_set:
				continue
			else:
				candidate = a_val + b_val
				if candidate > ans:
					ans = candidate
				break
	print(ans)

if __name__ == "__main__":
	main()