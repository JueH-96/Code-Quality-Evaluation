# YOUR CODE HERE
A, B, D = map(int, input().split())
terms = []
for i in range(A, B+1, D):
    terms.append(str(i))
print(' '.join(terms))