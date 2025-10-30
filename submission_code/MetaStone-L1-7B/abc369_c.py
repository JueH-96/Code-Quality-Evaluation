n = int(input())
a = list(map(int, input().split()))

if n == 0:
    print(0)
elif n == 1:
    print(1)
else:
    total = 1
    current_diff = None
    current_length = 1  # Tracks the length of the current AP ending at i

    for i in range(1, n):
        diff = a[i] - a[i-1]
        if i == 1:
            current_diff = diff
            current_length = 2
        else:
            if diff == current_diff:
                current_length += 1
            else:
                current_diff = diff
                current_length = 2
        total += current_length

    print(total)