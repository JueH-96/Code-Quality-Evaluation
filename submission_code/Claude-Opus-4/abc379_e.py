# YOUR CODE HERE
N = int(input())
S = input().strip()

total_sum = 0
sum_ending_here = 0

for i in range(N):
    digit = int(S[i])
    # All substrings ending at previous position, when extended by current digit
    # become: prev_substring * 10 + digit
    sum_ending_here = sum_ending_here * 10 + digit * (i + 1)
    total_sum += sum_ending_here

print(total_sum)