A = list(map(int, input().split()))
B = list(map(int, input().split()))

takahashi_score = sum(A)
aoki_score = sum(B)

diff = takahashi_score - aoki_score

if diff == 0:
    print(1)
else:
    print(diff + 1)