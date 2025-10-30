n, l, r = map(int, input().split())
a = list(range(1, n + 1))
l -= 1
r -= 1
sub_array = a[l:r+1]
reversed_sub_array = sub_array[::-1]
a[l:r+1] = reversed_sub_array
print(*a)