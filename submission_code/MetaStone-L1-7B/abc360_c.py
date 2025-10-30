n = int(input())
A = list(map(int, input().split()))
W = list(map(int, input().split()))

max_in_box = {}
for i in range(n):
    box = A[i]
    w = W[i]
    if box not in max_in_box or w > max_in_box[box]:
        max_in_box[box] = w

sum_max = sum(max_in_box.values())
total_w = sum(W)
print(total_w - sum_max)