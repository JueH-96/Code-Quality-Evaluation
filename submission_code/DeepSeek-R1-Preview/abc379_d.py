import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    Q = int(input[ptr])
    ptr += 1
    current_time = 0
    base_times = []
    left = 0
    for _ in range(Q):
        query_type = input[ptr]
        ptr += 1
        if query_type == '1':
            base_times.append(current_time)
        elif query_type == '2':
            T = int(input[ptr])
            ptr += 1
            current_time += T
        elif query_type == '3':
            H = int(input[ptr])
            ptr += 1
            X = current_time - H
            idx = bisect.bisect_right(base_times, X, left)
            print(idx - left)
            left = idx

if __name__ == '__main__':
    main()