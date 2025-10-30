import sys

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		n = int(data[index]); index += 1
		p = list(map(int, data[index:index+n]))
		index += n
		
		if p == list(range(1, n+1)):
			results.append("0")
			continue
			
		found_one = False
		for i in range(n):
			if p[i] != i+1:
				continue
			left_part = p[:i]
			if sorted(left_part) != list(range(1, i+1)):
				continue
			right_part = p[i+1:]
			if sorted(right_part) != list(range(i+2, n+1)):
				continue
			found_one = True
			break
			
		if found_one:
			results.append("1")
			continue
			
		is_descending = True
		for i in range(n):
			if p[i] != n - i:
				is_descending = False
				break
				
		if is_descending:
			results.append("3")
		else:
			results.append("2")
			
	print("
".join(results))

if __name__ == "__main__":
	main()