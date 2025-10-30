def main():
	D = input().strip()
	mapping = {
		'N': 'S',
		'S': 'N',
		'E': 'W',
		'W': 'E',
		'NE': 'SW',
		'SW': 'NE',
		'NW': 'SE',
		'SE': 'NW'
	}
	print(mapping[D])

if __name__ == "__main__":
	main()