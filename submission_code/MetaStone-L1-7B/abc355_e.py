# Read input
N, L, R = map(int, input().split())

len_sub = R - L + 1
current_L = L
total = 0

for i in range(N, -1, -1):
    mask = 1 << i
    if len_sub & mask:
        j = current_L // mask
        print(f"? {i} {j}")
        T = int(input().strip())
        total += T
        current_L += mask
        len_sub -= mask

print(total % 100)