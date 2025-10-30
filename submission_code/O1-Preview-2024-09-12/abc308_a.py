# YOUR CODE HERE
S = list(map(int, input().split()))
condition1 = all(S[i] <= S[i+1] for i in range(7))
condition2 = all(100 <= s <= 675 for s in S)
condition3 = all(s % 25 == 0 for s in S)
if condition1 and condition2 and condition3:
    print('Yes')
else:
    print('No')