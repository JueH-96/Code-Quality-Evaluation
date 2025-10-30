# YOUR CODE HERE
S = list(map(int, input().split()))

is_non_decreasing = all(S[i] <= S[i+1] for i in range(7))
is_in_range = all(100 <= S[i] <= 675 for i in range(8))
is_multiple_of_25 = all(S[i] % 25 == 0 for i in range(8))

if is_non_decreasing and is_in_range and is_multiple_of_25:
    print("Yes")
else:
    print("No")