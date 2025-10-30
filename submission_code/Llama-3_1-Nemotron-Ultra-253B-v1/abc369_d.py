n = int(input())
a = list(map(int, input().split()))

even = 0
odd = -float('inf')

for num in a:
    new_even = max(even, odd + 2 * num)
    new_odd = max(odd, even + num)
    even, odd = new_even, new_odd

print(max(even, odd))