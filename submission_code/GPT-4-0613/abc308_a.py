S = list(map(int, input().split()))
if all(100 <= s <= 675 and s % 25 == 0 for s in S) and all(S[i] <= S[i+1] for i in range(7)):
    print('Yes')
else:
    print('No')