import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print("No")
		return

	n = int(data[0])
	A = int(data[1])
	B = int(data[2])
	D_list = list(map(int, data[3:3+n]))
	T = A + B

	events = []

	for d_val in D_list:
		d = d_val % T
		L_i = (T - d) % T
		if L_i + A <= T:
			events.append((L_i, 1))
			events.append((L_i + A, -1))
		else:
			events.append((L_i, 1))
			events.append((T, -1))
			second_length = L_i + A - T
			if second_length > 0:
				events.append((0, 1))
				events.append((second_length, -1))
				
	events.sort(key=lambda x: x[0])
	
	current_count = 0
	prev = 0
	i = 0
	total_events = len(events)
	found = False
	
	while i < total_events:
		pos = events[i][0]
		if prev < pos and current_count == n:
			found = True
			break
			
		j = i
		while j < total_events and events[j][0] == pos:
			current_count += events[j][1]
			j += 1
			
		prev = pos
		i = j
		
	if not found and prev < T and current_count == n:
		found = True
		
	print("Yes" if found else "No")

if __name__ == "__main__":
	main()