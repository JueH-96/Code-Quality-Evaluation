import heapq
import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	it = iter(data)
	n = int(next(it))
	m = int(next(it))
	X1 = int(next(it))
	
	trains = []
	for i in range(m):
		a = int(next(it))
		b = int(next(it))
		s = int(next(it))
		t = int(next(it))
		trains.append((a, b, s, t))
	
	graph = [[] for _ in range(m)]
	city_events = {}
	
	for idx, (a, b, s, t) in enumerate(trains):
		if b not in city_events:
			city_events[b] = []
		city_events[b].append(('arrival', t, idx))
		if a not in city_events:
			city_events[a] = []
		city_events[a].append(('departure', s, idx))
	
	for city, events in city_events.items():
		arrivals = []
		departures = []
		for event in events:
			if event[0] == 'arrival':
				arrivals.append((event[1], event[2]))
			else:
				departures.append((event[1], event[2]))
		arrivals.sort()
		departures.sort()
		
		if not arrivals or not departures:
			continue
			
		j = 0
		for t_val, i_idx in arrivals:
			while j < len(departures) and departures[j][0] < t_val:
				j += 1
			for k in range(j, len(departures)):
				s_val, j_idx = departures[k]
				if s_val < t_val:
					continue
				weight = t_val - s_val
				graph[i_idx].append((j_idx, weight))
	
	INF = 10**18
	dist = [-INF] * m
	dist[0] = X1
	
	heap = [(-dist[0], 0)]
	
	while heap:
		d, node = heapq.heappop(heap)
		d = -d
		if d != dist[node]:
			continue
		for neighbor, weight in graph[node]:
			nd = d + weight
			if nd > dist[neighbor]:
				dist[neighbor] = nd
				heapq.heappush(heap, (-nd, neighbor))
	
	res = []
	for i in range(1, m):
		res.append(str(max(0, -dist[i])))
	
	print(" ".join(res))

if __name__ == '__main__':
	main()