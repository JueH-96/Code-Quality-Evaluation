import sys
import threading

def main():
    import sys
    input = sys.stdin.readline
    
    N, M = map(int, input().split())
    # For each possible L, store the minimum R among intervals starting at L
    INF = M + 1
    minR_at_L = [INF] * (M + 2)
    
    for _ in range(N):
        L, R = map(int, input().split())
        if R < minR_at_L[L]:
            minR_at_L[L] = R
    
    # bestR[l] = minimum R_i among all intervals with L_i >= l
    bestR_next = INF
    answer = 0
    
    # Traverse l from M down to 1
    for l in range(M, 0, -1):
        # incorporate intervals that start exactly at l
        if minR_at_L[l] < bestR_next:
            bestR_next = minR_at_L[l]
        # We must choose r so that r < bestR_next, and r >= l
        # Thus r runs from l to min(bestR_next-1, M)
        r_max = bestR_next - 1
        if r_max >= l:
            # count of integers in [l, r_max] is (r_max - l + 1)
            answer += (r_max - l + 1)
    
    print(answer)

if __name__ == "__main__":
    main()