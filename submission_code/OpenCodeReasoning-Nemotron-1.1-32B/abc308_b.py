def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	
	n, m = map(int, data[0].split())
	colors = data[1].split()
	special_colors = data[2].split()
	prices = list(map(int, data[3].split()))
	
	base_price = prices[0]
	special_dict = {}
	for i in range(m):
		special_dict[special_colors[i]] = prices[i+1]
	
	total = 0
	for c in colors:
		if c in special_dict:
			total += special_dict[c]
		else:
			total += base_price
			
	print(total)

if __name__ == "__main__":
	main()