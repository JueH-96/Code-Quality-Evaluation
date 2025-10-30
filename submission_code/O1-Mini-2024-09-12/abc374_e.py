# YOUR CODE HERE
import sys
import math

def readints():
    return list(map(int, sys.stdin.read().split()))

def minimal_cost(A, P, B, Q, C):
    if C == 0:
        return 0
    # s_i ranges from 0 to ceil(C / A)
    # Perform ternary search over s_i to find minimal cost
    s_low = 0
    s_high = (C + A -1) // A  # ceil(C/A)
    def cost(s):
        remaining = C - s * A
        if remaining <=0:
            t =0
        else:
            t = (remaining + B -1) // B
        return s * P + t * Q
    
    # Ternary search over integer domain
    while s_high - s_low > 3:
        m1 = s_low + (s_high - s_low) //3
        m2 = s_high - (s_high - s_low) //3
        cost1 = cost(m1)
        cost2 = cost(m2)
        if cost1 < cost2:
            s_high = m2
        else:
            s_low = m1
    min_cost = float('inf')
    for s in range(s_low, s_high+1):
        c = cost(s)
        if c < min_cost:
            min_cost = c
    return min_cost

def main():
    data = readints()
    idx=0
    N=X=0
    N, X = data[idx], data[idx+1]
    idx +=2
    processes = []
    for _ in range(N):
        A, P, B, Q = data[idx], data[idx+1], data[idx+2], data[idx+3]
        processes.append( (A, P, B, Q) )
        idx +=4
    left =0
    right = 10**14  # sufficiently large
    while left < right:
        mid = (left + right +1)//2
        total_cost =0
        for A,P,B,Q in processes:
            cost_i = minimal_cost(A,P,B,Q, mid)
            total_cost += cost_i
            if total_cost > X:
                break
        if total_cost <= X:
            left = mid
        else:
            right = mid -1
    print(left)

if __name__ == "__main__":
    main()