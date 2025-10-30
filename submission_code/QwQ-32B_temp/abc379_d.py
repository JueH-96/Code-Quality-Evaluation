import bisect
import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    Q = int(input[idx])
    idx += 1
    S = []
    current_total_time = 0
    ptr = 0

    for _ in range(Q):
        query_type = input[idx]
        if query_type == '1':
            S.append(current_total_time)
            idx += 1
        elif query_type == '2':
            T = int(input[idx+1])
            current_total_time += T
            idx += 2
        elif query_type == '3':
            H = int(input[idx+1])
            threshold = current_total_time - H
            idx_b = bisect.bisect_right(S, threshold)
            count = idx_b - ptr
            print(count)
            ptr = idx_b
            idx += 2

if __name__ == "__main__":
    main()