def main():
    import sys
    input = sys.stdin.readline
    MOD = 998244353

    N, Q = map(int, input().split())
    P = [0]*Q
    V = [0]*Q
    for i in range(Q):
        pi, vi = map(int, input().split())
        P[i] = pi
        V[i] = vi

    # reqC[i] = -1 if free, or 0/1 if fixed by some constraint
    reqC = [-1]*Q

    # Process all pairs (k,i) with k<i and V[k]>V[i]
    # Each such pair gives:
    #   if P[k] < P[i]:  c[k]=0, c[i]=1
    #   if P[k] > P[i]:  c[k]=1, c[i]=0
    #   if P[k] == P[i]: impossible -> answer = 0
    #
    # If we ever try to assign a variable to both 0 and 1, conflict -> 0.
    for i in range(Q):
        Pi = P[i]
        Vi = V[i]
        # iterate all earlier k
        for k in range(i):
            Vk = V[k]
            if Vk > Vi:
                Pk = P[k]
                # if equal P, no way to avoid overlap
                if Pk == Pi:
                    print(0)
                    return
                if Pk < Pi:
                    # require c[k]=0, c[i]=1
                    if reqC[k] == 1:
                        print(0)
                        return
                    reqC[k] = 0
                    if reqC[i] == 0:
                        print(0)
                        return
                    reqC[i] = 1
                else:
                    # Pk > Pi: require c[k]=1, c[i]=0
                    if reqC[k] == 0:
                        print(0)
                        return
                    reqC[k] = 1
                    if reqC[i] == 1:
                        print(0)
                        return
                    reqC[i] = 0

    # Count how many variables remain free (-1)
    free = 0
    for x in reqC:
        if x == -1:
            free += 1

    # Each free variable can be 0 or 1 independently
    ans = pow(2, free, MOD)
    print(ans)


if __name__ == "__main__":
    main()