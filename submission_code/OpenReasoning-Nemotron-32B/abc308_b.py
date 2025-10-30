def main():
	import sys
	data = sys.stdin.read().splitlines()
	first_line = data[0].split()
	n = int(first_line[0])
	m = int(first_line[1])
	
	colors = data[1].split()
	special_colors = data[2].split()
	prices = list(map(int, data[3].split()))
	
	price_dict = {}
	for i in range(m):
		color = special_colors[i]
		price_dict[color] = prices[i+1]
	
	total = 0
	base_price = prices[0]
	for c in colors:
		if c in price_dict:
			total += price_dict[c]
		else:
			total += base_price
			
	print(total)

if __name__ == "__main__":
	main()