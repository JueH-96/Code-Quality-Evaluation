n, *rest = map(int, open(0).read().split())
A = rest[:n]

prev_even = 0
prev_odd = float('-inf')

for a in A:
    current_even = max(prev_even, prev_odd + 2 * a)
    current_odd = max(prev_odd, prev_even + a)
    prev_even, prev_odd = current_even, current_odd

print(max(prev_even, prev_odd))