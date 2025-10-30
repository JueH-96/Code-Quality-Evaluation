import collections

def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	W = int(data[1])
	items = []
	index = 2
	for i in range(n):
		w = int(data[index])
		v = int(data[index+1])
		index += 2
		items.append((w, v))
	
	NEG_INF = -10**18
	dp_prev = [NEG_INF] * (W+1)
	dp_prev[0] = 0
	
	for w, v in items:
		dp_next = [NEG_INF] * (W+1)
		for r in range(w):
			T = (W - r) // w
			dq = collections.deque()
			for t in range(T+1):
				j = r + t * w
				a_new = t
				b_new = dp_prev[j] - t*t
				
				while len(dq) >= 2:
					a1, b1 = dq[-2]
					a2, b2 = dq[-1]
					left = (b2 - b1) * (a_new - a2)
					right = (b_new - b2) * (a2 - a1)
					if left <= right:
						dq.pop()
					else:
						break
				dq.append((a_new, b_new))
				
				m = 2*t - v
				while len(dq) >= 2:
					a1, b1 = dq[0]
					a2, b2 = dq[1]
					if m * (a1 - a2) <= (b2 - b1):
						dq.popleft()
					else:
						break
				a_opt, b_opt = dq[0]
				best = a_opt * m + b_opt
				candidate = -t*t + v*t + best
				if candidate > dp_next[j]:
					dp_next[j] = candidate
		dp_prev = dp_next
	
	ans = max(dp_prev)
	print(ans)

if __name__ == '__main__':
	main()