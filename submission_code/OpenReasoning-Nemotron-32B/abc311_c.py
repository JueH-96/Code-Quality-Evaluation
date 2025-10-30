import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	state = [0] * (n + 1)
	best_cycle = None
	
	for i in range(1, n + 1):
		if state[i] != 0:
			continue
		stack = []
		current = i
		while state[current] == 0:
			state[current] = 1
			stack.append(current)
			current = A[current - 1]
			
		if state[current] == 1:
			try:
				idx = stack.index(current)
			except ValueError:
				idx = 0
			cycle = stack[idx:]
			if best_cycle is None or len(cycle) > len(best_cycle):
				best_cycle = cycle
				
		for node in stack:
			state[node] = 2
			
	if best_cycle is None:
		print("2")
		print("1 2")
	else:
		min_node = min(best_cycle)
		min_index = best_cycle.index(min_node)
		rotated_cycle = best_cycle[min_index:] + best_cycle[:min_index]
		print(len(rotated_cycle))
		print(" ".join(map(str, rotated_cycle)))

if __name__ == '__main__':
	main()