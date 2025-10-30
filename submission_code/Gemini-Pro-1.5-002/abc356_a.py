# YOUR CODE HERE
n, l, r = map(int, input().split())
a = list(range(1, n + 1))
l -= 1
reversed_part = a[l:r]
reversed_part.reverse()
a = a[:l] + reversed_part + a[r:]
print(*a)