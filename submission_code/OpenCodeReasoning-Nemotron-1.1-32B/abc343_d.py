import sys
from collections import defaultdict

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	t = int(data[1])
	events = []
	index = 2
	for i in range(t):
		a = int(data[index])
		b = int(data[index + 1])
		index += 2
		events.append((a, b))
	
	scores = [0] * n
	freq = defaultdict(int)
	freq[0] = n
	
	for a, b in events:
		idx = a - 1
		old_score = scores[idx]
		new_score = old_score + b
		
		freq[old_score] -= 1
		if freq[old_score] == 0:
			del freq[old_score]
			
		freq[new_score] += 1
		scores[idx] = new_score
		
		print(len(freq))

if __name__ == "__main__":
	main()