import sys

def main():
	data = sys.stdin.read().splitlines()
	n = int(data[0].strip())
	cards = []
	for i in range(1, n + 1):
		a, c = map(int, data[i].split())
		cards.append((a, c, i))
	
	cards.sort(key=lambda x: x[0], reverse=True)
	
	min_so_far = 10**10
	remaining_indices = []
	
	for a, c, idx in cards:
		if c < min_so_far:
			remaining_indices.append(idx)
			min_so_far = c
			
	remaining_indices.sort()
	
	print(len(remaining_indices))
	if remaining_indices:
		print(" ".join(map(str, remaining_indices)))
	else:
		print()

if __name__ == "__main__":
	main()