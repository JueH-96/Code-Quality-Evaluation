# YOUR CODE HERE
n, l, r = map(int, input().split())
a = list(range(1, n + 1))
left = a[:l-1]
middle = a[l-1:r]
reversed_middle = middle[::-1]
right = a[r:]
result = left + reversed_middle + right
print(' '.join(map(str, result)))