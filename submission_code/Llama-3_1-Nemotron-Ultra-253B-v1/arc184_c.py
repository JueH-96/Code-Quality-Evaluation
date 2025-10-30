def count_parity(n):
    return bin(n).count('1') % 2

n = int(input())
a = list(map(int, input().split()))
even = 0
odd = 0
for num in a:
    if count_parity(num) % 2 == 0:
        even += 1
    else:
        odd += 1
print(max(even, odd))