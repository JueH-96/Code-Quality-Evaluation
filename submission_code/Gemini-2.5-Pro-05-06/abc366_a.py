# YOUR CODE HERE
N, T, A = map(int, input().split())

win_score = (N // 2) + 1

if T >= win_score:
    print("Yes")
elif A >= win_score:
    print("Yes")
else:
    print("No")