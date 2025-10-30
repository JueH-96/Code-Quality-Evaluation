import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	t = int(data[0])
	index = 1
	results = []
	
	for _ in range(t):
		S = data[index].strip()
		X = data[index+1].strip()
		Y = data[index+2].strip()
		index += 3
		
		a = X.count('0')
		b = len(X) - a
		c = Y.count('0')
		d = len(Y) - c
		lenS = len(S)
		
		if d == b:
			if (a - c) * lenS == 0:
				results.append("Yes")
			else:
				results.append("No")
		else:
			numerator = (a - c) * lenS
			denominator = d - b
			if numerator % denominator != 0:
				results.append("No")
			else:
				L = numerator // denominator
				if L < 0:
					results.append("No")
				else:
					i, j = 0, 0
					p, q = 0, 0
					T_dict = {}
					n = len(X)
					m = len(Y)
					valid = True
					
					while i < n or j < m:
						blockX = None
						lenX = 0
						if i < n:
							if X[i] == '0':
								blockX = 'S'
								lenX = lenS
							else:
								blockX = 'T'
								lenX = L
						else:
							break
							
						blockY = None
						lenY = 0
						if j < m:
							if Y[j] == '0':
								blockY = 'S'
								lenY = lenS
							else:
								blockY = 'T'
								lenY = L
						else:
							break
							
						if lenX == 0:
							i += 1
							p = 0
							continue
						if lenY == 0:
							j += 1
							q = 0
							continue
							
						segX = lenX - p
						segY = lenY - q
						min_len = min(segX, segY)
						if min_len == 0:
							if segX == 0:
								i += 1
								p = 0
							if segY == 0:
								j += 1
								q = 0
							continue
							
						if blockX == 'S' and blockY == 'S':
							if S[p:p+min_len] != S[q:q+min_len]:
								valid = False
								break
							p += min_len
							q += min_len
						elif blockX == 'S' and blockY == 'T':
							for k in range(min_len):
								idx = q + k
								if idx in T_dict:
									if T_dict[idx] != S[p+k]:
										valid = False
										break
								else:
									T_dict[idx] = S[p+k]
							else:
								p += min_len
								q += min_len
								if p == lenX:
									i += 1
									p = 0
								if q == lenY:
									j += 1
									q = 0
								continue
							break
						elif blockX == 'T' and blockY == 'S':
							for k in range(min_len):
								idx = p + k
								if idx in T_dict:
									if T_dict[idx] != S[q+k]:
										valid = False
										break
								else:
									T_dict[idx] = S[q+k]
							else:
								p += min_len
								q += min_len
								if p == lenX:
									i += 1
									p = 0
								if q == lenY:
									j += 1
									q = 0
								continue
							break
						else:
							for k in range(min_len):
								idx1 = p + k
								idx2 = q + k
								if idx1 in T_dict and idx2 in T_dict:
									if T_dict[idx1] != T_dict[idx2]:
										valid = False
										break
								elif idx1 in T_dict:
									T_dict[idx2] = T_dict[idx1]
								elif idx2 in T_dict:
									T_dict[idx1] = T_dict[idx2]
								else:
									T_dict[idx1] = 'a'
									T_dict[idx2] = 'a'
							else:
								p += min_len
								q += min_len
								if p == lenX:
									i += 1
									p = 0
								if q == lenY:
									j += 1
									q = 0
								continue
							break
							
					if valid:
						results.append("Yes")
					else:
						results.append("No")
						
	for res in results:
		print(res)

if __name__ == "__main__":
	main()