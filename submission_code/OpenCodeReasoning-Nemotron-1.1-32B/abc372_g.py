import sys

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		N = int(data[index]); index += 1
		constraints = []
		for i in range(N):
			a = int(data[index]); b = int(data[index+1]); c = int(data[index+2]); index += 3
			constraints.append((a, b, c))
		
		y_max = 10**18
		for (a, b, c) in constraints:
			if c - 1 < 0:
				temp = -1
			else:
				temp = (c - 1) // b
			if temp < y_max:
				y_max = temp
				
		if y_max < 1:
			results.append("0")
			continue
			
		low, high = 1, y_max + 1
		while low < high:
			mid = (low + high) // 2
			m_val = 10**18
			for (a, b, c) in constraints:
				num = c - 1 - b * mid
				if num < 0:
					temp = -10**18
				else:
					temp = num // a
				if temp < m_val:
					m_val = temp
			if m_val < 1:
				high = mid
			else:
				low = mid + 1
		y0 = low
		
		total = 0
		for y in range(1, y0):
			m_val = 10**18
			for (a, b, c) in constraints:
				num = c - 1 - b * y
				if num < 0:
					temp = -10**18
				else:
					temp = num // a
				if temp < m_val:
					m_val = temp
			if m_val < 1:
				break
			total += m_val
		results.append(str(total))
	
	print("
".join(results))

if __name__ == '__main__':
	main()