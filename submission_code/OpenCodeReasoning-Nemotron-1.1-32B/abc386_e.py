import sys
import itertools

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	k = int(data[1])
	a = list(map(int, data[2:2+n]))
	
	total = 0
	for x in a:
		total ^= x
		
	if k > n - k:
		k_prime = n - k
		best_candidate = -1
		for comb in itertools.combinations(a, k_prime):
			t = 0
			for x in comb:
				t ^= x
			candidate = total ^ t
			if candidate > best_candidate:
				best_candidate = candidate
		print(best_candidate)
	else:
		best = -1
		for comb in itertools.combinations(a, k):
			t = 0
			for x in comb:
				t ^= x
			if t > best:
				best = t
		print(best)

if __name__ == '__main__':
	main()