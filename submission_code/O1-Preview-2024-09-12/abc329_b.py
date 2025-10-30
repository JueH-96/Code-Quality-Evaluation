N = int(input())
A = list(map(int, input().split()))
max_A = max(A)
B = [x for x in A if x != max_A]
print(max(B))