import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	
	n, m = map(int, data[0].split())
	first_male_found = [False] * (n + 1)
	
	output_lines = []
	for i in range(1, m + 1):
		parts = data[i].split()
		if len(parts) < 2:
			continue
		family = int(parts[0])
		gender = parts[1].strip()
		if gender == 'M':
			if not first_male_found[family]:
				output_lines.append("Yes")
				first_male_found[family] = True
			else:
				output_lines.append("No")
		else:
			output_lines.append("No")
	
	for line in output_lines:
		print(line)

if __name__ == "__main__":
	main()