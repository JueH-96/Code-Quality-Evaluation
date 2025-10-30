N, T, A = map(int, input().split())
remaining_votes = N - (T + A)
if T > A + remaining_votes or A > T + remaining_votes:
    print("Yes")
else:
    print("No")