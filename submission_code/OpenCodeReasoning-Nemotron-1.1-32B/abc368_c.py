def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	H = list(map(int, data[1:1+n]))
	
	S = 1
	for health in H:
		s = S % 3
		if s == 0:
			damages = [0, 3, 4]
		elif s == 1:
			damages = [0, 1, 2]
		else:
			damages = [0, 1, 4]
			
		candidates = []
		for r in range(3):
			d = damages[r]
			if health <= d:
				k_candidate = r
			else:
				q = (health - d + 4) // 5
				k_candidate = 3 * q + r
			candidates.append(k_candidate)
			
		k_min = min(candidates)
		S += k_min
		
	print(S - 1)

if __name__ == '__main__':
	main()