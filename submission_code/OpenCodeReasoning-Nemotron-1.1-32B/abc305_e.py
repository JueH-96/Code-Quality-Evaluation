import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	it = iter(data)
	n = int(next(it))
	m = int(next(it))
	k = int(next(it))
	graph = [[] for _ in range(n + 1)]
	for _ in range(m):
		a = int(next(it))
		b = int(next(it))
		graph[a].append(b)
		graph[b].append(a)
	
	guards = []
	max_stamina = 0
	for _ in range(k):
		p = int(next(it))
		h = int(next(it))
		guards.append((p, h))
		if h > max_stamina:
			max_stamina = h
	
	best = [-1] * (n + 1)
	buckets = [[] for _ in range(max_stamina + 1)]
	
	for p, h in guards:
		if h > best[p]:
			best[p] = h
			if h <= max_stamina:
				buckets[h].append(p)
	
	for s in range(max_stamina, -1, -1):
		while buckets[s]:
			u = buckets[s].pop()
			if best[u] > s:
				continue
			for v in graph[u]:
				new_s = s - 1
				if new_s < 0:
					continue
				if new_s > best[v]:
					best[v] = new_s
					if new_s <= max_stamina:
						buckets[new_s].append(v)
	
	result = []
	for i in range(1, n + 1):
		if best[i] >= 0:
			result.append(i)
	
	print(len(result))
	if result:
		print(" ".join(map(str, result)))
	else:
		print()

if __name__ == '__main__':
	main()