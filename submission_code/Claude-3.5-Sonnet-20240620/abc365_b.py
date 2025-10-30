# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

max_value = max(A)
second_max = max(x for x in A if x != max_value)

print(A.index(second_max) + 1)