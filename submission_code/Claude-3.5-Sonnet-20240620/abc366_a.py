# YOUR CODE HERE
N, T, A = map(int, input().split())

remaining_votes = N - T - A
max_possible_T = T + remaining_votes
max_possible_A = A + remaining_votes

if max_possible_T > N // 2 and T > A:
    print("Yes")
elif max_possible_A > N // 2 and A > T:
    print("Yes")
else:
    print("No")