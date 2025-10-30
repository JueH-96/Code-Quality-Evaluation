n = int(input())
a_list = list(map(int, input().split()))

even = 0
odd = -float('inf')

for a in a_list:
    new_even = max(even, odd + 2 * a)
    new_odd = max(odd, even + a)
    even, odd = new_even, new_odd

print(max(even, odd))