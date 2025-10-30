import sys
from collections import defaultdict

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	events = []
	index = 1
	for i in range(n):
		t = int(data[index])
		x = int(data[index+1])
		index += 2
		events.append((t, x))
	
	potions_by_type = defaultdict(list)
	monsters_by_type = defaultdict(list)
	
	for i, (t, x) in enumerate(events):
		if t == 1:
			potions_by_type[x].append(i)
		else:
			monsters_by_type[x].append(i)
			
	for x in list(monsters_by_type.keys()):
		A = potions_by_type.get(x, [])
		B = monsters_by_type[x]
		A.sort()
		B.sort()
		if len(A) < len(B):
			print(-1)
			return
		for j in range(len(B)):
			if A[j] > B[j]:
				print(-1)
				return
				
	pickup = [False] * n
	
	for x in monsters_by_type:
		A = sorted(potions_by_type[x])
		B = sorted(monsters_by_type[x])
		B_desc = sorted(B, reverse=True)
		i = len(A) - 1
		for b in B_desc:
			while i >= 0 and A[i] > b:
				i -= 1
			if i < 0:
				break
			pickup[A[i]] = True
			i -= 1
			
	current_inv = 0
	max_inv = 0
	for i in range(n):
		t, x = events[i]
		if t == 1:
			if pickup[i]:
				current_inv += 1
		else:
			current_inv -= 1
		if current_inv > max_inv:
			max_inv = current_inv
			
	actions_output = []
	for i in range(n):
		if events[i][0] == 1:
			actions_output.append('1' if pickup[i] else '0')
			
	print(max_inv)
	if actions_output:
		print(" ".join(actions_output))
	else:
		print()

if __name__ == "__main__":
	main()