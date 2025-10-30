N, T, A = map(int, input().split())
winning_threshold = (N // 2) + 1
if T >= winning_threshold or A >= winning_threshold:
    print("Yes")
else:
    print("No")