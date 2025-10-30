import sys

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		K = int(data[index])
		sx = int(data[index+1])
		sy = int(data[index+2])
		tx = int(data[index+3])
		ty = int(data[index+4])
		index += 5
		
		def get_tile(x, y):
			i = x // K
			j = y // K
			if i % 2 == j % 2:
				k_val = y - j * K
			else:
				k_val = x - i * K
			return (i, j, k_val)
		
		i1, j1, k1 = get_tile(sx, sy)
		i2, j2, k2 = get_tile(tx, ty)
		
		di = abs(i1 - i2)
		dj = abs(j1 - j2)
		dk = abs(k1 - k2)
		ans = di + dj + dk
		results.append(str(ans))
	
	print("
".join(results))

if __name__ == '__main__':
	main()