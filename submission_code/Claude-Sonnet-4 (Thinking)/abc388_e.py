n = int(input())
a = list(map(int, input().split()))

small = 0  # Pointer for the smaller mochi
large = 1  # Pointer for the larger mochi
count = 0

while small < n and large < n:
    # Try to pair a[small] with a[large]
    if a[large] >= 2 * a[small]:
        # Valid pair
        count += 1
        small += 1
        large += 1
    else:
        # a[large] is too small to pair with a[small]
        # Try the next larger mochi
        large += 1

print(count)