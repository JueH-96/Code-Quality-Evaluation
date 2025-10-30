def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		return
	n, r = map(int, data[0].split())
	contests = []
	for i in range(1, n + 1):
		d, a = map(int, data[i].split())
		contests.append((d, a))
	
	current_rating = r
	for d, a in contests:
		if d == 1:
			if 1600 <= current_rating <= 2799:
				current_rating += a
		elif d == 2:
			if 1200 <= current_rating <= 2399:
				current_rating += a
	print(current_rating)

if __name__ == '__main__':
	main()