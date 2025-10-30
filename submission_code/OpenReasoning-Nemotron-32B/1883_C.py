import sys

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		n = int(data[index]); index += 1
		k = int(data[index]); index += 1
		a = list(map(int, data[index:index+n]))
		index += n
		
		if k == 2:
			if any(x % 2 == 0 for x in a):
				results.append("0")
			else:
				results.append("1")
				
		elif k == 3:
			if any(x % 3 == 0 for x in a):
				results.append("0")
			else:
				min_ops = min(3 - (x % 3) for x in a)
				results.append(str(min_ops))
				
		elif k == 4:
			if any(x % 4 == 0 for x in a):
				results.append("0")
			else:
				count_even = sum(1 for x in a if x % 2 == 0)
				if count_even >= 2:
					results.append("0")
				elif count_even == 1:
					results.append("1")
				else:
					min_div4 = 10**9
					for x in a:
						r = x % 4
						if r == 0:
							cost = 0
						else:
							cost = 4 - r
						if cost < min_div4:
							min_div4 = cost
					ans = min(min_div4, 2)
					results.append(str(ans))
					
		elif k == 5:
			if any(x % 5 == 0 for x in a):
				results.append("0")
			else:
				min_ops = min(5 - (x % 5) for x in a)
				results.append(str(min_ops))
				
	print("
".join(results))

if __name__ == "__main__":
	main()