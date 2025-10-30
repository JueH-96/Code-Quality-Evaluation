# YOUR CODE HERE
import sys
import threading

def main():
    import bisect
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A_list = list(map(int, N_and_rest[1:]))

    N = len(A_list)
    A = A_list
    if N == 0:
        print(0)
        return

    A.sort()
    prefix_counts = [0] * (N +1)
    prefix_sums = [0] * (N +1)
    total_sum = 0

    for i in range(N):
        prefix_counts[i+1] = prefix_counts[i] + 1
        prefix_sums[i+1] = prefix_sums[i] + A[i]
        total_sum += A[i]

    min_A = A[0]
    max_A = A[-1]

    left = min_A
    right = max_A

    def cost(x):
        # Compute total moves to make array elements x or x+1
        idx_x = bisect.bisect_left(A, x)
        idx_x1 = bisect.bisect_right(A, x+1)

        # Elements less than x
        count_less = idx_x
        sum_less = prefix_sums[idx_x]
        total_increase = x * count_less - sum_less

        # Elements greater than x+1
        count_greater = N - idx_x1
        sum_greater = total_sum - prefix_sums[idx_x1]
        total_decrease = sum_greater - (x +1) * count_greater

        total_moves = total_increase + total_decrease
        return total_moves

    l = min_A
    r = max_A
    while l <= r:
        if r - l <= 3:
            min_cost = None
            for x in range(l, r+1):
                c = cost(x)
                if min_cost is None or c < min_cost:
                    min_cost = c
            print(int(min_cost))
            return
        mid1 = l + (r - l) // 3
        mid2 = r - (r - l) // 3
        cost1 = cost(mid1)
        cost2 = cost(mid2)
        if cost1 < cost2:
            r = mid2 -1
        else:
            l = mid1 +1

if __name__ == "__main__":
    threading.Thread(target=main).start()