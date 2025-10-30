import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	S = data[0].strip()
	n = len(S)
	if not data[1:]:
		return
	Q = int(data[1])
	if len(data) < 3:
		K_list = []
	else:
		K_list = list(map(int, data[2].split()))
	
	res = []
	for K in K_list:
		if K <= n:
			res.append(S[K-1])
		else:
			total = n
			m = 0
			while total < K:
				total *= 2
				m += 1
			
			inv = 0
			current_K = K
			current_total = total
			for _ in range(m):
				half = current_total // 2
				if current_K <= half:
					current_total = half
				else:
					current_K -= half
					inv = 1 - inv
					current_total = half
					
			c = S[current_K-1]
			if inv:
				if c.islower():
					c = c.upper()
				else:
					c = c.lower()
			res.append(c)
			
	print(" ".join(res))

if __name__ == '__main__':
	main()