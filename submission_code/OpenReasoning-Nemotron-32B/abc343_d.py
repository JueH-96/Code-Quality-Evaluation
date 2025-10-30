import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	t = int(data[1])
	events = []
	index = 2
	for i in range(t):
		a = int(data[index])
		b = int(data[index+1])
		index += 2
		events.append((a, b))
	
	scores = [0] * (n + 1)
	freq = {0: n}
	ans = []
	
	for a, b in events:
		old_score = scores[a]
		new_score = old_score + b
		
		if freq[old_score] == 1:
			del freq[old_score]
		else:
			freq[old_score] -= 1
		
		scores[a] = new_score
		
		if new_score in freq:
			freq[new_score] += 1
		else:
			freq[new_score] = 1
		
		ans.append(str(len(freq)))
	
	print("
".join(ans))

if __name__ == '__main__':
	main()