def main():
	s = input().strip()
	parts = s.split('|')
	blocks = parts[1:-1]
	result_list = [str(len(block)) for block in blocks]
	print(" ".join(result_list))

if __name__ == '__main__':
	main()