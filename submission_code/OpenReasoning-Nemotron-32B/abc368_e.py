import sys
from collections import defaultdict

def update_fenw(fenw, index, value):
	n = len(fenw) - 1
	while index <= n:
		if value > fenw[index]:
			fenw[index] = value
		index += index & -index

def query_fenw(fenw, index):
	res = -10**18
	while index:
		if fenw[index] > res:
			res = fenw[index]
		index -= index & -index
	return res

def main():
	data = sys.stdin.read().split()
	if not data: 
		return
	it = iter(data)
	N = int(next(it)); M = int(next(it)); X1 = int(next(it))
	trains = []
	for i in range(M):
		A = int(next(it)); B = int(next(it)); S = int(next(it)); T = int(next(it))
		trains.append((A, B, S, T))
	
	events = []
	for i, (A, B, S, T) in enumerate(trains):
		events.append((S, 1, i, A))
		events.append((T, 0, i, B))
	
	events.sort(key=lambda x: (x[0], x[1]))
	
	city_events = defaultdict(list)
	for time, typ, i, city in events:
		city_events[city].append(time)
	
	city_ds = {}
	for city, times in city_events.items():
		unique_times = sorted(set(times))
		comp = {}
		for idx, t_val in enumerate(unique_times, start=1):
			comp[t_val] = idx
		n = len(unique_times)
		fenw_tree = [-10**18] * (n+1)
		city_ds[city] = (comp, fenw_tree, n)
	
	F = [0] * M
	F[0] = X1
	
	for event in events:
		time, typ, i, city = event
		if typ == 0:
			if city in city_ds:
				comp, fenw, n = city_ds[city]
				pos = comp[time]
				update_fenw(fenw, pos, time + F[i])
		else:
			if i == 0:
				continue
			if city in city_ds:
				comp, fenw, n = city_ds[city]
				pos = comp[time]
				max_val = query_fenw(fenw, pos)
				F[i] = max(0, max_val - time)
			else:
				F[i] = 0
				
	print(" ".join(str(x) for x in F[1:]))

if __name__ == "__main__":
	main()