import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	healths = list(map(int, data[1:1+n]))
	
	total_attacks = 0
	for health in healths:
		r = total_attacks % 3
		low, high = 0, health
		while low < high:
			mid = (low + high) // 2
			count = (total_attacks + mid) // 3 - total_attacks // 3
			damage = mid + 2 * count
			if damage >= health:
				high = mid
			else:
				low = mid + 1
		total_attacks += low
		
	print(total_attacks)

if __name__ == "__main__":
	main()