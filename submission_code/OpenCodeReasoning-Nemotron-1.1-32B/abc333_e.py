import sys
from collections import defaultdict

def main():
	data = sys.stdin.read().split()
	if not data:
		print(-1)
		return
	
	n = int(data[0])
	events = []
	index = 1
	for i in range(n):
		t = int(data[index])
		x = int(data[index + 1])
		index += 2
		events.append((t, x))
	
	potion_events = defaultdict(list)
	monster_events = defaultdict(list)
	
	for i in range(n):
		t, x = events[i]
		if t == 1:
			potion_events[x].append(i)
		else:
			monster_events[x].append(i)
	
	for x in monster_events:
		if x not in potion_events:
			print(-1)
			return
		P = sorted(potion_events[x])
		M = sorted(monster_events[x])
		if len(M) > len(P):
			print(-1)
			return
		for i in range(len(M)):
			if P[i] > M[i]:
				print(-1)
				return
				
	pickup = [False] * n
	
	for x in monster_events:
		P = sorted(potion_events[x])
		M = sorted(monster_events[x])
		j = len(P) - 1
		for i in range(len(M) - 1, -1, -1):
			m = M[i]
			while j >= 0 and P[j] >= m:
				j -= 1
			if j < 0:
				print(-1)
				return
			pickup[P[j]] = True
			j -= 1
			
	current = 0
	max_inv = 0
	for i in range(n):
		t, x = events[i]
		if t == 1:
			if pickup[i]:
				current += 1
		else:
			current -= 1
		if current > max_inv:
			max_inv = current
			
	print(max_inv)
	res = []
	for i in range(n):
		if events[i][0] == 1:
			res.append('1' if pickup[i] else '0')
	print(" ".join(res))

if __name__ == '__main__':
	main()