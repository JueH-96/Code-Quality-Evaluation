import math
from collections import deque
from fractions import Fraction

def floor_sum(n, m, a, b):
	ans = 0
	if a < 0:
		a2 = a % m
		if a2 < 0:
			a2 += m
		cnt = (a2 - a) // m
		ans -= n * (n - 1) // 2 * cnt
		a = a2
	if b < 0:
		b2 = b % m
		if b2 < 0:
			b2 += m
		cnt = (b2 - b) // m
		ans -= n * cnt
		b = b2
	while True:
		if a >= m:
			cnt = a // m
			ans += n * (n - 1) // 2 * cnt
			a %= m
		if b >= m:
			cnt = b // m
			ans += n * cnt
			b %= m
		y_max = a * n + b
		if y_max < m:
			break
		n = y_max // m
		b = y_max % m
		m, a = a, m
		ans += (n - 1) * n // 2 * (a // m)
		a %= m
	return ans

def line_intersection(line1, line2):
	a1, b1, c1 = line1
	a2, b2, c2 = line2
	denom = a1 * b2 - a2 * b1
	if denom == 0:
		return None
	x = Fraction(b1 * c2 - b2 * c1, denom)
	y = Fraction(a2 * c1 - a1 * c2, denom)
	return (x, y)

def is_point_in_half_plane(point, line):
	x, y = point
	a, b, c = line
	return a * x + b * y + c <= 0

def half_plane_intersection(lines):
	lines = [tuple(line) for line in lines]
	unique_lines = {}
	for line in lines:
		a, b, c = line
		if a == 0 and b == 0:
			if c < 0:
				return []
			else:
				continue
		g = math.gcd(math.gcd(a, b), c)
		a //= g
		b //= g
		c //= g
		if a < 0 or (a == 0 and b < 0):
			a, b, c = -a, -b, -c
		key = (a, b)
		if key not in unique_lines:
			unique_lines[key] = (a, b, c)
		else:
			a0, b0, c0 = unique_lines[key]
			if c < c0:
				unique_lines[key] = (a, b, c)
	lines = list(unique_lines.values())
	if not lines:
		return []
	lines.sort(key=lambda line: (math.atan2(line[1], line[0])))
	def cross_angle(line1, line2):
		a1, b1, c1 = line1
		a2, b2, c2 = line2
		return a1 * b2 - a2 * b1
	sorted_lines = []
	for i in range(len(lines)):
		if i > 0 and cross_angle(lines[i], lines[i-1]) == 0:
			continue
		sorted_lines.append(lines[i])
	lines = sorted_lines
	dq = deque()
	for i in range(len(lines)):
		while len(dq) >= 2:
			p = line_intersection(dq[-1], dq[-2])
			if p is None:
				dq.pop()
				continue
			if is_point_in_half_plane(p, lines[i]):
				break
			dq.pop()
		while len(dq) >= 2:
			p = line_intersection(dq[0], dq[1])
			if p is None:
				dq.popleft()
				continue
			if is_point_in_half_plane(p, lines[i]):
				break
			dq.popleft()
		if not dq:
			dq.append(lines[i])
		else:
			if line_intersection(dq[-1], lines[i]) is None:
				dq.pop()
			dq.append(lines[i])
	while len(dq) >= 3:
		p = line_intersection(dq[-1], dq[-2])
		if p is None:
			dq.pop()
			continue
		if is_point_in_half_plane(p, dq[0]):
			break
		dq.pop()
	while len(dq) >= 3:
		p = line_intersection(dq[0], dq[1])
		if p is None:
			dq.popleft()
			continue
		if is_point_in_half_plane(p, dq[-1]):
			break
		dq.popleft()
	if len(dq) < 3:
		return []
	polygon = []
	dq_list = list(dq)
	for i in range(len(dq_list)):
		line1 = dq_list[i]
		line2 = dq_list[(i+1) % len(dq_list)]
		p = line_intersection(line1, line2)
		if p is None:
			continue
		polygon.append(p)
	return polygon

def main():
	import sys
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		n = int(data[index]); index += 1
		lines = [(-1, 0, 1), (0, -1, 1)]
		for i in range(n):
			a = int(data[index]); b = int(data[index+1]); c = int(data[index+2]); index += 3
			lines.append((a, b, -(c-1)))
		polygon = half_plane_intersection(lines)
		if not polygon or len(polygon) < 3:
			results.append("0")
			continue
		vertices = set()
		for p in polygon:
			vertices.add(p)
		x_to_ys = {}
		for (x, y) in vertices:
			if x not in x_to_ys:
				x_to_ys[x] = []
			x_to_ys[x].append(y)
		xs = sorted(x_to_ys.keys())
		for x in xs:
			y_min = min(x_to_ys[x])
			y_max = max(x_to_ys[x])
			x_to_ys[x] = (y_min, y_max)
		total_count = 0
		for i in range(len(xs)-1):
			x_i = xs[i]
			x_j = xs[i+1]
			y_min_i, y_max_i = x_to_ys[x_i]
			y_min_j, y_max_j = x_to_ys[x_j]
			a_upper = y_max_j - y_max_i
			b_upper = y_max_i * x_j - y_max_j * x_i
			m_upper = x_j - x_i
			a_lower = y_min_j - y_min_i
			b_lower = y_min_i * x_j - y_min_j * x_i
			m_lower = x_j - x_i
			a0 = math.ceil(x_i)
			b0 = math.floor(x_j)
			if a0 > b0:
				continue
			L = b0 - a0 + 1
			part_upper = floor_sum(L, m_upper, a_upper, a_upper * a0 + b_upper)
			part_lower = floor_sum(L, m_lower, -a_lower, -a_lower * a0 - b_lower)
			count_interval = L + part_upper - part_lower
			total_count += count_interval
		results.append(str(total_count))
	print("
".join(results))

if __name__ == "__main__":
	main()