import sys

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		n = int(data[index])
		index += 1
		a = list(map(int, data[index:index+n]))
		index += n
		
		prefix = [0] * (n+1)
		for i in range(1, n+1):
			prefix[i] = prefix[i-1] + a[i-1]
			
		divisors = set()
		i = 1
		while i * i <= n:
			if n % i == 0:
				divisors.add(i)
				divisors.add(n // i)
			i += 1
			
		ans = 0
		for k in divisors:
			t_val = n // k
			min_val = float('inf')
			max_val = 0
			for i in range(t_val):
				start_idx = i * k
				end_idx = start_idx + k
				s = prefix[end_idx] - prefix[start_idx]
				if s < min_val:
					min_val = s
				if s > max_val:
					max_val = s
			diff = max_val - min_val
			if diff > ans:
				ans = diff
				
		results.append(str(ans))
	
	sys.stdout.write("
".join(results))

if __name__ == "__main__":
	main()