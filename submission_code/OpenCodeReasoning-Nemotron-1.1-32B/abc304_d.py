import bisect
import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	W = int(data[0])
	H = int(data[1])
	N = int(data[2])
	strawberries = []
	index = 3
	for i in range(N):
		p = int(data[index])
		q = int(data[index+1])
		index += 2
		strawberries.append((p, q))
	
	A = int(data[index])
	index += 1
	vertical_cuts = list(map(int, data[index:index+A]))
	index += A
	
	B = int(data[index])
	index += 1
	horizontal_cuts = list(map(int, data[index:index+B]))
	
	total_pieces = (A + 1) * (B + 1)
	count_dict = {}
	
	for p, q in strawberries:
		i = bisect.bisect_right(vertical_cuts, p)
		j = bisect.bisect_right(horizontal_cuts, q)
		key = (i, j)
		count_dict[key] = count_dict.get(key, 0) + 1
		
	distinct_pieces = len(count_dict)
	if distinct_pieces < total_pieces:
		min_straw = 0
	else:
		min_straw = min(count_dict.values())
		
	max_straw = max(count_dict.values()) if count_dict else 0
	
	print(f"{min_straw} {max_straw}")

if __name__ == '__main__':
	main()