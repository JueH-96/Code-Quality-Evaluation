N, L = map(int, input().split())
A = list(map(int, input().split()))
print(sum(1 for a in A if a >= L))