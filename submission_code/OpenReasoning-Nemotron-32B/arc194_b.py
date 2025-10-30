import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	p = list(map(int, data[1:1+n]))
	
	pos = [0] * (n + 1)
	for idx in range(n):
		val = p[idx]
		pos[val] = idx + 1
		
	bit = [0] * (n + 1)
	
	def update(index, delta):
		while index <= n:
			bit[index] += delta
			index += index & -index
			
	def query(index):
		s = 0
		while index:
			s += bit[index]
			index -= index & -index
		return s
		
	total_cost = 0
	for i in range(1, n + 1):
		c = pos[i]
		moved_count = query(c - 1)
		current_position = c - moved_count
		cost_i = current_position * (current_position - 1) // 2
		total_cost += cost_i
		update(c, 1)
		
	print(total_cost)

if __name__ == '__main__':
	main()