import sys
from collections import deque

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		print(-1)
		return
	
	N = int(data[0].strip())
	S = data[1].strip()
	T = data[2].strip()
	
	if sorted(S) != sorted(T):
		print(-1)
		return
		
	L = N + 2
	start = S + ".."
	target = T + ".."
	
	if start == target:
		print(0)
		return
		
	visited = set()
	queue = deque()
	queue.append((start, 0))
	visited.add(start)
	
	while queue:
		state, steps = queue.popleft()
		if state == target:
			print(steps)
			return
			
		e = state.find("..")
		if e == -1:
			continue
			
		for j in range(L-1):
			if state[j] == '.' or state[j+1] == '.':
				continue
				
			lst = list(state)
			lst[e] = state[j]
			lst[e+1] = state[j+1]
			lst[j] = '.'
			lst[j+1] = '.'
			new_state = ''.join(lst)
			
			if new_state not in visited:
				visited.add(new_state)
				queue.append((new_state, steps+1))
				
	print(-1)

if __name__ == "__main__":
	main()