# YOUR CODE HERE
N, T, A = map(int, input().split())

remaining = N - T - A
votes_needed_to_win = (N + 1) // 2

# Election is decided if:
# 1. Takahashi already won, or Aoki can't catch up even with all remaining votes
# 2. Aoki already won, or Takahashi can't catch up even with all remaining votes

if T >= votes_needed_to_win or A + remaining < votes_needed_to_win:
    print("Yes")
elif A >= votes_needed_to_win or T + remaining < votes_needed_to_win:
    print("Yes")
else:
    print("No")