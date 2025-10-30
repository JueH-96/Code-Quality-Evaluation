import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # right_counts[x] = count of occurrences of x in positions > current j (initially all)
    # left_counts[x] = count of occurrences of x in positions < current j (initially 0)
    # sum_total = sum over x of left_counts[x] * right_counts[x]
    # When moving j from 1..N:
    #   1) Remove A[j] from right_counts: right_counts[x] -= 1
    #      => sum_total decreases by left_counts[x]
    #   2) The number of valid triples with this j as middle is:
    #         sum_total - left_counts[A[j]] * right_counts[A[j]]
    #   3) Insert A[j] into left_counts: left_counts[x] += 1
    #      => sum_total increases by right_counts[x]

    # Since A[i] <= N, we size counts to N+1
    right_counts = [0] * (N + 1)
    left_counts = [0] * (N + 1)

    # fill right_counts
    for x in A:
        right_counts[x] += 1

    sum_total = 0
    result = 0

    # iterate j from 1..N (0-based in code)
    for j in range(N):
        x = A[j]
        # 1) remove from right
        right_counts[x] -= 1
        # adjust sum_total
        sum_total -= left_counts[x]

        # 2) count triples with this j
        # sum_total = sum_{v} left[v]*right[v]
        # we must exclude v == x where A_i == A_j
        result += sum_total - left_counts[x] * right_counts[x]

        # 3) add to left
        sum_total += right_counts[x]
        left_counts[x] += 1

    # print answer
    print(result)

if __name__ == "__main__":
    main()