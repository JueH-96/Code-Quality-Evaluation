import sys
input = sys.stdin.read

def find_sequence():
    data = input().strip().split()
    N = int(data[0])
    pairs = []
    index = 1

    for _ in range(N):
        L = int(data[index])
        R = int(data[index + 1])
        pairs.append((L, R))
        index += 2

    left_sum = sum(L for L, R in pairs)
    right_sum = sum(R for L, R in pairs)

    if left_sum > 0 or right_sum < 0:
        print("No")
        return

    current_sum = 0
    sequence = []

    for i in range(N):
        L, R = pairs[i]
        if L <= -current_sum <= R:
            sequence.append(-current_sum)
            current_sum = 0
        else:
            sequence.append(R)
            current_sum += R

    if current_sum == 0:
        print("Yes")
        print(" ".join(map(str, sequence)))
    else:
        print("No")

find_sequence()