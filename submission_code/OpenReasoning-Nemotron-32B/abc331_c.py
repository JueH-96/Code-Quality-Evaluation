import bisect

def main():
	n = int(input().strip())
	arr = list(map(int, input().split()))
	
	total_sum = sum(arr)
	sorted_arr = sorted(arr)
	prefix = [0] * (n + 1)
	for i in range(1, n + 1):
		prefix[i] = prefix[i - 1] + sorted_arr[i - 1]
	
	res = []
	for x in arr:
		idx = bisect.bisect_right(sorted_arr, x)
		s = total_sum - prefix[idx]
		res.append(str(s))
	
	print(" ".join(res))

if __name__ == "__main__":
	main()