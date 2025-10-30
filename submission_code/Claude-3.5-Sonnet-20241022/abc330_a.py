N, L = map(int, input().split())
A = list(map(int, input().split()))
print(sum(1 for score in A if score >= L))