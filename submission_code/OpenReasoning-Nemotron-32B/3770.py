class Solution:
	def generateString(self, str1: str, str2: str) -> str:
		n = len(str1)
		m = len(str2)
		total_len = n + m - 1
		res = [None] * total_len
		
		for i in range(n):
			if str1[i] == 'T':
				for j in range(m):
					pos = i + j
					if res[pos] is not None and res[pos] != str2[j]:
						return ""
					res[pos] = str2[j]
		
		for i in range(n):
			if str1[i] == 'F':
				all_set = True
				match = True
				for j in range(m):
					pos = i + j
					if res[pos] is None:
						all_set = False
						break
					if res[pos] != str2[j]:
						match = False
						break
				if all_set and match:
					return ""
		
		constraints_by_position = [[] for _ in range(total_len)]
		for i in range(n):
			if str1[i] == 'F':
				for j in range(i, i + m):
					if j < total_len:
						constraints_by_position[j].append(i)
		
		mismatch = [False] * n
		set_count = [0] * n
		
		for i in range(n):
			if str1[i] == 'F':
				for j in range(m):
					pos = i + j
					if res[pos] is not None:
						if res[pos] != str2[j]:
							mismatch[i] = True
							break
						else:
							set_count[i] += 1
		
		for j in range(total_len):
			if res[j] is not None:
				continue
			found = False
			for c in "abcdefghijklmnopqrstuvwxyz":
				valid = True
				for i in constraints_by_position[j]:
					if mismatch[i]:
						continue
					idx2 = j - i
					if c != str2[idx2]:
						continue
					else:
						if set_count[i] + 1 == m:
							valid = False
							break
				if valid:
					res[j] = c
					for i in constraints_by_position[j]:
						if mismatch[i]:
							continue
						idx2 = j - i
						if c != str2[idx2]:
							mismatch[i] = True
						else:
							set_count[i] += 1
					found = True
					break
			if not found:
				return ""
		
		return ''.join(res)