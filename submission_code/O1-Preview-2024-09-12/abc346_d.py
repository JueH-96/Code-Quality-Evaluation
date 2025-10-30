# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import bisect
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    C = list(map(int, sys.stdin.readline().split()))
    min_total_cost = None
    for b in ['0', '1']:
        N = len(S)
        E = []
        F = []
        a = []
        b_list = []
        delta_cost = [0]*N
        total_cost = 0
        for i in range(N):
            expected_bit = b if i % 2 == 0 else ('1' if b == '0' else '0')
            flipped_bit = '1' if expected_bit == '0' else '0'
            E.append(expected_bit)
            F.append(flipped_bit)
            if S[i] == expected_bit:
                a.append(1)
                total_cost += 0  # No cost
            else:
                a.append(0)
                total_cost += C[i]  # Cost is C[i]
            if S[i] == flipped_bit:
                b_list.append(1)
            else:
                b_list.append(0)
            delta_cost[i] = (b_list[i] - a[i]) * C[i]
        delta_cost_cumsum = [0]*(N+2)
        for i in range(N-1, -1, -1):
            delta_cost_cumsum[i] = delta_cost[i] + delta_cost_cumsum[i+1]
        for k in range(N-1):
            total_cost_k = total_cost + delta_cost_cumsum[k+1]
            if min_total_cost is None or total_cost_k < min_total_cost:
                min_total_cost = total_cost_k
    print(min_total_cost)

threading.Thread(target=main).start()