import sys
from collections import deque

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(-1)
		return
	n = int(data[0].strip())
	s = data[1].strip()
	t = data[2].strip()
	
	if sorted(s) != sorted(t):
		print(-1)
		return
		
	start = s + '..'
	target = t + '..'
	
	if start == target:
		print(0)
		return
		
	visited = set()
	queue = deque()
	queue.append((start, 0))
	visited.add(start)
	
	while queue:
		state, cost = queue.popleft()
		if state == target:
			print(cost)
			return
			
		e_index = -1
		for i in range(len(state)-1):
			if state[i] == '.' and state[i+1] == '.':
				e_index = i
				break
				
		for i in range(len(state)-1):
			if state[i] != '.' and state[i+1] != '.':
				lst = list(state)
				lst[e_index] = state[i]
				lst[e_index+1] = state[i+1]
				lst[i] = '.'
				lst[i+1] = '.'
				new_state = ''.join(lst)
				if new_state not in visited:
					visited.add(new_state)
					queue.append((new_state, cost+1))
					
	print(-1)

if __name__ == "__main__":
	main()