import sys
import threading

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Find the maximum value in A to size our arrays
    maxA = max(A)

    # sum_vals[v] will hold the total sum of all occurrences of value v
    sum_vals = [0] * (maxA + 1)
    for v in A:
        sum_vals[v] += v

    # Build suffix sums: suffix_sum[v] = sum of sum_vals[w] for all w >= v
    # We only need up to index maxA+1 so that suffix_sum[a+1] is safe for all a in A
    suffix_sum = [0] * (maxA + 2)
    for v in range(maxA, -1, -1):
        suffix_sum[v] = suffix_sum[v + 1] + sum_vals[v]

    # For each A_i, the answer is sum of all values strictly greater than A_i,
    # i.e., suffix_sum[A_i + 1]
    out = []
    for v in A:
        out.append(str(suffix_sum[v + 1]))

    sys.stdout.write(" ".join(out))

if __name__ == "__main__":
    main()