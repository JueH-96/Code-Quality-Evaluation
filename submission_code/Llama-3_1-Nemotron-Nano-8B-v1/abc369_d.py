n, *rest = map(int, open(0).read().split())
a = rest[:n]

prev_even = 0
prev_odd = float('-inf')

for num in a:
    new_even = max(prev_even, prev_odd + 2 * num)
    new_odd = max(prev_odd, prev_even + num)
    prev_even, prev_odd = new_even, new_odd

print(max(prev_even, prev_odd))