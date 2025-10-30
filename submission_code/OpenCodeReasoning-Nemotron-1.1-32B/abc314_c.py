n, m = map(int, input().split())
s = input().strip()
c_list = list(map(int, input().split()))

groups = [[] for _ in range(m+1)]
for i in range(n):
	color_val = c_list[i]
	groups[color_val].append(i)

s_list = list(s)

for color in range(1, m+1):
	indices = groups[color]
	k = len(indices)
	if k <= 1:
		continue
	last_char = s_list[indices[-1]]
	for j in range(k-1, 0, -1):
		s_list[indices[j]] = s_list[indices[j-1]]
	s_list[indices[0]] = last_char

print(''.join(s_list))