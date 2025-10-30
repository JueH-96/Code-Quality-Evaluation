import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	n = int(data[0].strip())
	S = data[1].strip()
	
	if n == 0:
		print("")
		return
		
	next_arr = [0] * n
	prev_arr = [0] * n
	for i in range(n):
		if i == n-1:
			next_arr[i] = -1
		else:
			next_arr[i] = i+1
		prev_arr[i] = i-1
		
	head = 0
	stack = []
	current = head
	
	while current != -1:
		c = S[current]
		if c == '(':
			stack.append(current)
			current = next_arr[current]
		elif c == ')':
			if stack:
				l = stack.pop()
				r = current
				l_prev = prev_arr[l]
				r_next = next_arr[r]
				
				if l_prev != -1:
					next_arr[l_prev] = r_next
				else:
					head = r_next
					
				if r_next != -1:
					prev_arr[r_next] = l_prev
					
				current = r_next
			else:
				current = next_arr[current]
		else:
			current = next_arr[current]
			
	res = []
	cur = head
	while cur != -1:
		res.append(S[cur])
		cur = next_arr[cur]
		
	print(''.join(res))

if __name__ == "__main__":
	main()