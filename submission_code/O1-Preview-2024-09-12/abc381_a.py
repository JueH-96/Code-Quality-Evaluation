# YOUR CODE HERE
N = int(input())
S = input()

if N % 2 == 0:
    print('No')
else:
    mid = N // 2
    if S[mid] != '/':
        print('No')
    elif mid >= 1 and any(c != '1' for c in S[:mid]):
        print('No')
    elif mid + 1 < N and any(c != '2' for c in S[mid+1:]):
        print('No')
    else:
        print('Yes')