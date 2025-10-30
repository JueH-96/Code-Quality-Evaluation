suspects = {1, 2, 3}
A, B = map(int, input().split())
if A == B:
    eliminated = {A}
else:
    eliminated = {A, B}
remaining = suspects - eliminated
if len(remaining) == 1:
    print(remaining.pop())
else:
    print(-1)