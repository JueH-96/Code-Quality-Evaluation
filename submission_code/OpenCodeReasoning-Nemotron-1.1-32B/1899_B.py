import sys

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		n = int(data[index]); index += 1
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
			
		best_diff = 0
		for k in divisors:
			t_val = n // k
			min_val = float('inf')
			max_val = float('-inf')
			for i in range(t_val):
				start_index = i * k
				end_index = start_index + k
				seg_sum = prefix[end_index] - prefix[start_index]
				if seg_sum < min_val:
					min_val = seg_sum
				if seg_sum > max_val:
					max_val = seg_sum
			diff = max_val - min_val
			if diff > best_diff:
				best_diff = diff
				
		results.append(str(best_diff))
	
	print("
".join(results))

if __name__ == '__main__':
	main()