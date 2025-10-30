import sys

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		n = int(data[index]); index += 1
		k = int(data[index]); index += 1
		cakes = []
		for i in range(n):
			x = int(data[index]); y = int(data[index+1]); z = int(data[index+2]); index += 3
			cakes.append((x, y, z))
		
		if k == 0:
			results.append("0")
			continue
			
		max_vals = [max(x, y, z) for (x, y, z) in cakes]
		indices = list(range(n))
		indices.sort(key=lambda i: max_vals[i], reverse=True)
		m = min(n, 3 * k)
		reduced = [cakes[i] for i in indices[:m]]
		
		if m <= 16:
			dp = [-10**18] * (1 << m)
			dp[0] = 0
			for mask in range(1 << m):
				if dp[mask] < 0:
					continue
				i = 0
				while i < m and (mask >> i) & 1:
					i += 1
				if i >= m:
					continue
				for j in range(i + 1, m):
					if (mask >> j) & 1:
						continue
					x1, y1, z1 = reduced[i]
					x2, y2, z2 = reduced[j]
					w = max(x1 + x2, y1 + y2, z1 + z2)
					new_mask = mask | (1 << i) | (1 << j)
					if dp[new_mask] < dp[mask] + w:
						dp[new_mask] = dp[mask] + w
			ans = -10**18
			for mask in range(1 << m):
				if bin(mask).count('1') == 2 * k:
					if dp[mask] > ans:
						ans = dp[mask]
			if ans == -10**18:
				ans = 0
			results.append(str(ans))
		else:
			reduced.sort(key=lambda cake: cake[0] + cake[1] + cake[2], reverse=True)
			total = 0
			for i in range(0, 2 * k, 2):
				x1, y1, z1 = reduced[i]
				x2, y2, z2 = reduced[i + 1]
				w = max(x1 + x2, y1 + y2, z1 + z2)
				total += w
			results.append(str(total))
	
	print("
".join(results))

if __name__ == "__main__":
	main()