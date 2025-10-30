n = int(input())
a_list = list(map(int, input().split()))

prev_even = 0
prev_odd = float('-inf')

for a in a_list:
    current_even = max(prev_even, prev_odd + 2 * a)
    current_odd = max(prev_odd, prev_even + a)
    prev_even, prev_odd = current_even, current_odd

print(max(prev_even, prev_odd))