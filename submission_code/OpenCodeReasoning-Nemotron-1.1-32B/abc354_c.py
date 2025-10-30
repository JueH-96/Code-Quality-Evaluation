import sys

def main():
	data = sys.stdin.read().splitlines()
	n = int(data[0].strip())
	cards = []
	for i in range(1, n+1):
		a, c = map(int, data[i].split())
		cards.append((a, c, i))
	
	cards.sort(key=lambda x: x[0], reverse=True)
	
	min_cost = 10**10
	kept = []
	
	for a, c, idx in cards:
		if c < min_cost:
			kept.append(idx)
			min_cost = c
			
	kept.sort()
	
	print(len(kept))
	if kept:
		print(" ".join(map(str, kept)))

if __name__ == "__main__":
	main()