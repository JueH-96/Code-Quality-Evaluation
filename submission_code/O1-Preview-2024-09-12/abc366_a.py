# YOUR CODE HERE
N, T, A = map(int, input().split())
R = N - (T + A)  # Remaining votes

if T > A:
    max_votes_Aoki = A + R
    if max_votes_Aoki < T or (max_votes_Aoki == T and R == 0):
        print("Yes")
    else:
        print("No")
elif A > T:
    max_votes_Takahashi = T + R
    if max_votes_Takahashi < A or (max_votes_Takahashi == A and R == 0):
        print("Yes")
    else:
        print("No")
else:  # T == A
    print("No")