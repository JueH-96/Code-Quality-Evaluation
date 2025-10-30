n = int(input())
A = list(map(int, input().split()))
m = int(input())
B = list(map(int, input().split()))
l = int(input())
C = list(map(int, input().split()))
q = int(input())
X = list(map(int, input().split()))

s_set = set()
for a in A:
	for b in B:
		for c in C:
			s_set.add(a + b + c)

for x in X:
	print("Yes" if x in s_set else "No")