def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		return
	
	n, m = map(int, data[0].split())
	has_male = [False] * (n + 1)
	
	output_lines = []
	for i in range(1, m + 1):
		parts = data[i].split()
		family = int(parts[0])
		gender = parts[1]
		if gender == 'M':
			if not has_male[family]:
				output_lines.append("Yes")
				has_male[family] = True
			else:
				output_lines.append("No")
		else:
			output_lines.append("No")
	
	for line in output_lines:
		print(line)

if __name__ == "__main__":
	main()