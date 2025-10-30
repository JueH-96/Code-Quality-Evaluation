import bisect

def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	X = list(map(int, data[1:1+n]))
	q = int(data[1+n])
	tasks = []
	index = 1 + n + 1
	for i in range(q):
		t = int(data[index])
		g = int(data[index+1])
		index += 2
		tasks.append((t, g))
	
	cur = [0] * (n+1)
	s = []
	pos_to_index = {}
	
	for i in range(1, n+1):
		cur[i] = X[i-1]
		pos_to_index[X[i-1]] = i
		
	s = sorted(X)
	ans = 0

	def query_count(low, high):
		left = bisect.bisect_left(s, low)
		right = bisect.bisect_right(s, high)
		return right - left

	for task in tasks:
		t, g = task
		a = cur[t]
		if a == g:
			continue
			
		idx_a = bisect.bisect_left(s, a)
		del s[idx_a]
		del pos_to_index[a]
		
		L = min(a, g)
		R = max(a, g)
		
		left_idx = bisect.bisect_left(s, L)
		right_idx = bisect.bisect_right(s, R)
		k = right_idx - left_idx
		blockers = s[left_idx:left_idx+k] if k > 0 else []
		sum_blockers = sum(blockers) if k > 0 else 0
		
		extra = 0
		new_positions = []
		
		if k > 0:
			left_low = L - k
			left_high = L - 1
			left_count = query_count(left_low, left_high)
			left_free = (left_count == 0)
			
			right_low = R + 1
			right_high = R + k
			right_count = query_count(right_low, right_high)
			right_free = (right_count == 0)
			
			INF = 10**30
			if left_free:
				cost_left = sum_blockers - (k * L - (k*(k+1)//2)
			else:
				cost_left = INF
				
			if right_free:
				cost_right = (k * R + (k*(k+1)//2) - sum_blockers
			else:
				cost_right = INF
				
			if cost_left < cost_right:
				extra = cost_left
				new_positions = [L - 1 - i for i in range(k)]
			else:
				extra = cost_right
				new_positions = [R + 1 + i for i in range(k)]
				
			for i in range(k):
				p = blockers[i]
				new_p = new_positions[i]
				idx_person = pos_to_index[p]
				cur[idx_person] = new_p
				del pos_to_index[p]
				pos_to_index[new_p] = idx_person
				
			del s[left_idx:left_idx+k]
			for np in new_positions:
				bisect.insort(s, np)
				
		cur[t] = g
		pos_to_index[g] = t
		bisect.insort(s, g)
		
		ans += abs(a - g) + extra
		
	print(ans)

if __name__ == "__main__":
	main()