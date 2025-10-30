import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    B_dict = dict()
    W_list = []

    for _ in range(M):
        X = int(input[ptr])
        ptr += 1
        Y = int(input[ptr])
        ptr += 1
        C = input[ptr]
        ptr += 1
        if C == 'B':
            if X in B_dict:
                if Y > B_dict[X]:
                    B_dict[X] = Y
            else:
                B_dict[X] = Y
        else:
            W_list.append((X, Y))

    events = []
    events_x = []
    if W_list:
        W_list.sort()
        current_min = float('inf')
        for X, Y in W_list:
            y_minus_1 = Y - 1
            if y_minus_1 < current_min:
                current_min = y_minus_1
            events.append((X, current_min))
            events_x.append(X)

    for X in B_dict:
        lower = B_dict[X]
        if not events:
            upper = N
        else:
            idx = bisect.bisect_right(events_x, X) - 1
            if idx >= 0:
                upper = events[idx][1]
            else:
                upper = N
        if lower > upper:
            print("No")
            return

    print("Yes")

if __name__ == '__main__':
    main()