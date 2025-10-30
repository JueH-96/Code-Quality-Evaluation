# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))

multiples_of_K = [a // K for a in A if a % K == 0]
multiples_of_K.sort()

print(' '.join(map(str, multiples_of_K)))