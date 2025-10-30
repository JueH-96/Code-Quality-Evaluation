# YOUR CODE HERE
N = int(input())
output = ''
for i in range(N+1):
    s_i = '-'
    for j in range(1, 10):
        if N % j == 0:
            n_over_j = N // j
            if i % n_over_j == 0:
                s_i = str(j)
                break
    output += s_i
print(output)