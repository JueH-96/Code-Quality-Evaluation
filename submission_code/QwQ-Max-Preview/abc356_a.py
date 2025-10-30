n, l, r = map(int, input().split())
a = list(range(1, n + 1))
reversed_part = a[l-1:r][::-1]
new_a = a[:l-1] + reversed_part + a[r:]
print(' '.join(map(str, new_a)))