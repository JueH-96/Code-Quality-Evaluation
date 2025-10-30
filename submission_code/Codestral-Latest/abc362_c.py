import sys
input = sys.stdin.read
data = input().split()

n = int(data[0])
lr = [(int(data[i]), int(data[i + 1])) for i in range(1, len(data), 2)]

# Check if the sum of L_i and R_i is zero
if sum(l[0] for l in lr) <= 0 <= sum(r[1] for r in lr):
    # Find a valid sequence
    x = []
    total_sum = 0
    for i in range(n):
        l, r = lr[i]
        if total_sum + l <= 0 <= total_sum + r:
            if total_sum + l == 0:
                x.append(l)
            else:
                x.append(-total_sum)
            total_sum = 0
        else:
            x.append(l)
            total_sum += l

    if total_sum == 0:
        print("Yes")
        print(" ".join(map(str, x)))
    else:
        print("No")
else:
    print("No")