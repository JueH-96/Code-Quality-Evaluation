# YOUR CODE HERE
import sys, math
def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    N,K,*rest = map(int, sys.stdin.read().split())
    A = rest[:K]
    count = [2]*(N+1)
    for a in A:
        count[a] -=1
    singletons = []
    for i in range(1,N+1):
        if count[i]==1:
            singletons.append(i)
    S = len(singletons)
    INF = 1 << 60
    dp0 = [INF]*(S+1)
    dp1 = [INF]*(S+1)
    dp0[0] = 0
    dp1[0] = INF
    for i in range(1,S+1):
        if i ==1:
            dp0[i] = INF
            dp1[i] = dp0[i-1]
        else:
            dp0[i] = dp0[i-2] + abs(singletons[i-1] - singletons[i-2])
            dp1[i] = min(dp0[i-1], dp1[i-2] + abs(singletons[i-1] - singletons[i-2]))
    if S %2 ==0:
        singleton_sum = dp0[S]
    else:
        singleton_sum = dp1[S]
    # Now, count the total sum from pairs that can be made within counts
    # All such pairs have zero weirdness
    print(singleton_sum)

if __name__ == "__main__":
    main()