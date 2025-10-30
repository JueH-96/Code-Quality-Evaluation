n = int(input().strip())
s = input().strip()
q = int(input().strip())
operations = []
for _ in range(q):
	c, d = input().split()
	operations.append((c, d))

mapping = {chr(ord('a') + i): chr(ord('a') + i) for i in range(26)}

for i in range(q-1, -1, -1):
	c, d = operations[i]
	mapping[c] = mapping[d]

result = ''.join(mapping[char] for char in s)
print(result)