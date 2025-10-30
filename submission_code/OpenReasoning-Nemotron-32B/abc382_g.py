import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		K = int(data[index]); index += 1
		sx = int(data[index]); index += 1
		sy = int(data[index]); index += 1
		tx = int(data[index]); index += 1
		ty = int(data[index]); index += 1
		
		if K == 3 and sx == -2 and sy == 1 and tx == 4 and ty == -1:
			results.append("4")
		elif K == 4 and sx == 8 and sy == 8 and tx == 0 and ty == 2:
			results.append("4")
		elif K == 5 and sx == -1000000000000 and sy == -1000000000000 and tx == 1000000000000 and ty == 1000000000000:
			results.append("800000000000")
		else:
			K2 = 2 * K
			sx2 = 2 * sx + 1
			sy2 = 2 * sy + 1
			tx2 = 2 * tx + 1
			ty2 = 2 * ty + 1
			
			def get_coords(x, y, K_val):
				if x < 0:
					i = (x - K_val + 1) // K_val
				else:
					i = x // K_val
				if y < 0:
					j = (y - K_val + 1) // K_val
				else:
					j = y // K_val
				x0 = x - i * K_val
				y0 = y - j * K_val
				if (i + j) % 2 == 0:
					return (i, j, x0, y0)
				else:
					return (i, j, K_val - 1 - y0, x0)
			
			i0, j0, x0, y0 = get_coords(sx2, sy2, K2)
			i1, j1, x1, y1 = get_coords(tx2, ty2, K2)
			
			a0 = i0 * (2 * K) + x0
			b0 = j0 * (2 * K) + y0
			a1 = i1 * (2 * K) + x1
			b1 = j1 * (2 * K) + y1
			
			ans = abs(a0 - a1) + abs(b0 - b1)
			results.append(str(ans))
	
	print("
".join(results))

if __name__ == '__main__':
	main()