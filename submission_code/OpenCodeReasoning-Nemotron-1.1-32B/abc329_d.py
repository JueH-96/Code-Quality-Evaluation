import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	votes = list(map(int, data[2:2+m]))
	
	count = [0] * (n + 1)
	current_max_count = 0
	current_winner = 0
	
	out_lines = []
	for a in votes:
		count[a] += 1
		if count[a] > current_max_count:
			current_max_count = count[a]
			current_winner = a
		elif count[a] == current_max_count:
			if a < current_winner:
				current_winner = a
		out_lines.append(str(current_winner))
	
	sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
	main()