import sys

N, L, R = map(int, sys.stdin.readline().split())
sum_mod = 0
current = L

while current <= R:
    x = R - current + 1
    # Compute i1
    if current == 0:
        i1 = N
    else:
        temp = current
        i1 = 0
        while temp % 2 == 0:
            i1 += 1
            temp //= 2
    # Compute i2
    i2 = x.bit_length() - 1
    i = min(i1, i2)
    block_size = 1 << i
    j = current >> i
    # Send query
    print(f"? {i} {j}")
    sys.stdout.flush()
    T = int(sys.stdin.readline().strip())
    if T == -1:
        exit()
    sum_mod = (sum_mod + T) % 100
    current += block_size

print(f"! {sum_mod}")