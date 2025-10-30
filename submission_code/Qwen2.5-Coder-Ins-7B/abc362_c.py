import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
pairs = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N)]

# Calculate the total sum of the ranges
total_sum = sum(R - L for L, R in pairs)

# If the total sum is not zero, it's impossible to find such a sequence
if total_sum != 0:
    print("No")
else:
    # Try to construct the sequence
    X = []
    current_sum = 0
    for L, R in pairs:
        if L <= 0 <= R:
            X.append(0)
            current_sum += 0
        elif L < 0:
            X.append(L)
            current_sum += L
        else:
            X.append(R)
            current_sum += R

    # Adjust the sequence to ensure the sum is zero
    if current_sum != 0:
        X[-1] += -current_sum

    print("Yes")
    print(" ".join(map(str, X)))