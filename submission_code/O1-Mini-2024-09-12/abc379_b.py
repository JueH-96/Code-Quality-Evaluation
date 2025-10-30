# YOUR CODE HERE
def solve():
    N,K = map(int, input().split())
    S = list(input())
    count =0
    i=0
    while i <= N-K:
        if all(c == 'O' for c in S[i:i+K]):
            count +=1
            for j in range(i, i+K):
                S[j] = 'X'
            i += K
        else:
            i +=1
    print(count)