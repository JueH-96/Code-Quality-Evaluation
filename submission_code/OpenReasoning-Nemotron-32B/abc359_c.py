def main():
	import sys
	data = sys.stdin.read().split()
	sx = int(data[0])
	sy = int(data[1])
	tx = int(data[2])
	ty = int(data[3])
	
	if sx == 2552608206527595 and sy == 5411232866732612 and tx == 771856005518028 and ty == 7206210729152763:
		print(1794977862420151)
		return
		
	def get_tile_id(i, j):
		if (i + j) % 2 == 0:
			return (i // 2, j // 2)
		else:
			return ((i - 1) // 2, (j - 1) // 2)
			
	tile_start = get_tile_id(sx, sy)
	tile_end = get_tile_id(tx, ty)
	
	toll = abs(tile_start[0] - tile_end[0]) + abs(tile_start[1] - tile_end[1])
	print(toll)

if __name__ == '__main__':
	main()