import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	k = int(data[1])
	A = list(map(int, data[2:2+n]))
	
	if n == 0:
		return
		
	M = max(A)
	freq = [0] * (M + 1)
	for a in A:
		if a <= M:
			freq[a] += 1
			
	count = [0] * (M + 1)
	for d in range(1, M + 1):
		total = 0
		j = d
		while j <= M:
			total += freq[j]
			j += d
		count[d] = total
		
	ans_arr = [0] * (M + 1)
	for d in range(1, M + 1):
		if count[d] >= k:
			j = d
			while j <= M:
				if d > ans_arr[j]:
					ans_arr[j] = d
				j += d
					
	for a in A:
		print(ans_arr[a])

if __name__ == '__main__':
	main()