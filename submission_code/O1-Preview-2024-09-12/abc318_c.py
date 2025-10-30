# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N, D, P = map(int, sys.stdin.readline().split())
    F = list(map(int, sys.stdin.readline().split()))
    F.sort(reverse=True)
    prefix_sum = [0] * (N + 1)
    total_regular_fare = 0
    for i in range(N):
        total_regular_fare += F[i]
        prefix_sum[i + 1] = prefix_sum[i] + F[i]

    min_cost = total_regular_fare
    passes_used = 0
    batches_needed = 0
    N = len(F)
    while passes_used < N:
        K_next = min(passes_used + D, N)
        total_cost_if_buy = (batches_needed + 1) * P + (total_regular_fare - prefix_sum[K_next])
        if total_cost_if_buy < min_cost:
            min_cost = total_cost_if_buy
            batches_needed +=1
            passes_used = K_next
        else:
            break

    print(min_cost)

if __name__ == "__main__":
    threading.Thread(target=main).start()