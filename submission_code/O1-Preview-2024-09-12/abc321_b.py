# YOUR CODE HERE
N_and_X = input().split()
N = int(N_and_X[0])
X = int(N_and_X[1])
A = list(map(int, input().split()))

sum_A = sum(A)
min_A = min(A)
max_A = max(A)

ans = -1

# First interval: s in [0, min_A)
G1 = sum_A - max_A
if G1 >= X:
    ans = 0
else:
    # Second interval: s in [min_A, max_A)
    c = sum_A - min_A - max_A
    s_min = max(min_A, X - c)
    if s_min < max_A and s_min <= 100:
        ans = s_min
    else:
        # Third interval: s in [max_A, 100]
        G3 = sum_A - min_A
        if G3 >= X:
            ans = max_A
        else:
            ans = -1

print(int(ans))