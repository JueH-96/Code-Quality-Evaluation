N, K = map(int, input().split())
A = list(map(int, input().split()))

multiples = []
for num in A:
  if num % K == 0:
    multiples.append(num // K)

multiples.sort()
print(*multiples)