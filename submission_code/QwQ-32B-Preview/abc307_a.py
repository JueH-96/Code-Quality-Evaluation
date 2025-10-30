N_and_steps = []
try:
    while True:
        line = input()
        if not line:
            break
        N_and_steps.append(line)
except EOFError:
    pass

if len(N_and_steps) < 1:
    N = 0
    A = []
elif len(N_and_steps) == 1:
    N = int(N_and_steps[0])
    A = []
else:
    N = int(N_and_steps[0])
    A = list(map(int, N_and_steps[1].split()))

# Ensure we have exactly 7*N elements
if len(A) < 7 * N:
    additional_elements = 7 * N - len(A)
    for _ in range(additional_elements):
        A.append(0)

# Calculate weekly sums
weekly_sums = []
for i in range(N):
    start = i * 7
    end = start + 7
    week_sum = sum(A[start:end])
    weekly_sums.append(week_sum)

# Print the results
print(' '.join(map(str, weekly_sums)))