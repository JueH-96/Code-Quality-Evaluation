import sys

def main():
    n, q = map(int, sys.stdin.readline().split())
    x = list(map(int, sys.stdin.readline().split()))
    A = [0] * (n + 1)
    in_S = set()
    start_time = dict()
    prefix_sum = [0]
    for i in range(1, q + 1):
        xi = x[i - 1]
        if xi in in_S:
            start = start_time[xi]
            end = i - 1
            sum_contribution = prefix_sum[end] - prefix_sum[start - 1]
            A[xi] += sum_contribution
            in_S.remove(xi)
            del start_time[xi]
        else:
            in_S.add(xi)
            start_time[xi] = i
        current_size = len(in_S)
        prefix_sum.append(prefix_sum[-1] + current_size)
    for xi in in_S:
        start = start_time[xi]
        end = q
        sum_contribution = prefix_sum[end] - prefix_sum[start - 1]
        A[xi] += sum_contribution
    print(' '.join(map(str, A[1:n + 1])))

if __name__ == "__main__":
    main()