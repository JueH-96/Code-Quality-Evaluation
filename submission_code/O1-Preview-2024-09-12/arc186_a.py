# YOUR CODE HERE
N,Q=map(int,input().split())
K_list=[int(input()) for _ in range(Q)]
for K in K_list:
    if N==2:
        if K==4:
            print('Yes')
        else:
            print('No')
    elif N==3:
        if K==0 or K==9:
            print('Yes')
        else:
            print('No')
    else:
        if 0<=K<=N*N:
            print('Yes')
        else:
            print('No')