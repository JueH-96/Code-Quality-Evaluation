import math
import bisect

def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	
	it = iter(data)
	N = int(next(it))
	M = int(next(it))
	C = int(next(it))
	K = int(next(it))
	A = [int(next(it)) for _ in range(N)]
	
	g = math.gcd(C, M)
	T = M // g if g != 0 else 1
	m_prime = M // g if g != 0 else 0
	c_prime = C // g if g != 0 else 0
	
	S = sorted(set(A))
	events_set = set()
	events_set.add(0)
	events_set.add(M)
	for s in S:
		events_set.add(M - s)
	events = sorted(events_set)
	
	segments = []
	for i in range(len(events) - 1):
		a = events[i]
		b = events[i + 1]
		if a == b:
			continue
		min1 = None
		if S and S[0] <= M - b:
			min1 = S[0]
		min2 = None
		pos = bisect.bisect_left(S, M - a)
		if pos < len(S):
			min2 = S[pos]
		if min2 is not None:
			c_val = min2 - M
		else:
			c_val = min1
		segments.append((a, b, c_val))
	
	def sum_in_range(low, high, g, segments):
		total = 0
		for a, b, c_val in segments:
			seg_low = max(a, low)
			seg_high = min(b - 1, high)
			if seg_low > seg_high:
				continue
			j_low = (seg_low + g - 1) // g
			j_high = seg_high // g
			if j_low > j_high:
				continue
			n = j_high - j_low + 1
			sum_j = n * (j_low + j_high) // 2
			total += c_val * n + g * sum_j
		return total

	period_sum = sum_in_range(0, M - 1, g, segments) if g != 0 else 0
	Q = K // T
	R = K % T
	rem_sum = sum_in_range(0, (R - 1) * g, g, segments) if R > 0 else 0
	total_sum = Q * period_sum + rem_sum
	print(total_sum)

if __name__ == '__main__':
	main()