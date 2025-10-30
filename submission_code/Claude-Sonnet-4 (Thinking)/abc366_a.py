N, T, A = map(int, input().split())

votes_needed_to_win = (N + 1) // 2

if T >= votes_needed_to_win or A >= votes_needed_to_win:
    print("Yes")
else:
    print("No")