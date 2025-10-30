import sys

def main():
	data = sys.stdin.read().splitlines()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		S = data[index].strip()
		index += 1
		X = data[index].strip()
		index += 1
		Y = data[index].strip()
		index += 1
		
		n = len(S)
		a = X.count('0')
		b = len(X) - a
		c = Y.count('0')
		d = len(Y) - c
		
		if a == c:
			results.append("Yes")
			continue
			
		if (a - c) * (d - b) <= 0:
			results.append("No")
			continue
			
		if (a - c) * n % (d - b) != 0:
			results.append("No")
			continue
			
		k = (a - c) * n // (d - b)
		if k <= 0:
			results.append("No")
			continue
			
		starts_x = [0]
		for char in X:
			if char == '0':
				starts_x.append(starts_x[-1] + n)
			else:
				starts_x.append(starts_x[-1] + k)
				
		starts_y = [0]
		for char in Y:
			if char == '0':
				starts_y.append(starts_y[-1] + n)
			else:
				starts_y.append(starts_y[-1] + k)
				
		if starts_x[-1] != starts_y[-1]:
			results.append("No")
			continue
			
		constraints = []
		
		for i in range(len(X)):
			if X[i] == '0':
				sx = starts_x[i]
				ex = starts_x[i+1] - 1
				for j in range(len(Y)):
					if Y[j] == '1':
						sy = starts_y[j]
						ey = starts_y[j+1] - 1
						low = max(sx, sy)
						high = min(ex, ey)
						if low <= high:
							start_in_S = low - sx
							end_in_S = high - sx
							substr = S[start_in_S: end_in_S+1]
							start_in_T = low - sy
							end_in_T = high - sy
							constraints.append((start_in_T, end_in_T, substr))
							
		for i in range(len(Y)):
			if Y[i] == '0':
				sy = starts_y[i]
				ey = starts_y[i+1] - 1
				for j in range(len(X)):
					if X[j] == '1':
						sx = starts_x[j]
						ex = starts_x[j+1] - 1
						low = max(sx, sy)
						high = min(ex, ey)
						if low <= high:
							start_in_S = low - sy
							end_in_S = high - sy
							substr = S[start_in_S: end_in_S+1]
							start_in_T = low - sx
							end_in_T = high - sx
							constraints.append((start_in_T, end_in_T, substr))
							
		T_dict = {}
		valid = True
		for (start_t, end_t, s) in constraints:
			for offset in range(start_t, end_t+1):
				char_required = s[offset - start_t]
				if offset in T_dict:
					if T_dict[offset] != char_required:
						valid = False
						break
				else:
					T_dict[offset] = char_required
			if not valid:
				break
				
		if not valid:
			results.append("No")
			continue
			
		for offset in T_dict:
			if offset < 0 or offset >= k:
				valid = False
				break
				
		results.append("Yes" if valid else "No")
		
	print("
".join(results))

if __name__ == "__main__":
	main()