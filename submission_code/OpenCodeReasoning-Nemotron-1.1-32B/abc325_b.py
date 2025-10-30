def main():
	import sys
	data = sys.stdin.read().splitlines()
	n = int(data[0].strip())
	events = []
	index = 1
	for i in range(n):
		parts = data[index].split()
		index += 1
		if not parts:
			continue
		W = int(parts[0])
		X = int(parts[1])
		L = 9 - X
		R_plus1 = 18 - X
		events.append((L, W))
		events.append((R_plus1, -W))
	
	events.sort(key=lambda x: x[0])
	
	current_sum = 0
	max_sum = 0
	i = 0
	total_events = len(events)
	while i < total_events:
		time_val = events[i][0]
		temp = 0
		while i < total_events and events[i][0] == time_val:
			temp += events[i][1]
			i += 1
		current_sum += temp
		if current_sum > max_sum:
			max_sum = current_sum
			
	print(max_sum)

if __name__ == "__main__":
	main()