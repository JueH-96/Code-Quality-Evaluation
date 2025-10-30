import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        A = list(map(int, input[ptr:ptr+N]))
        ptr += N
        prefix = [0] * (N + 1)
        for i in range(N):
            prefix[i+1] = prefix[i] + A[i]
        T_total = prefix[-1]
        possible = True
        s_prev_prev = 0
        s_prev = prefix[1]
        if s_prev > T_total:
            possible = False
        for i in range(2, N+1):
            required = 2 * s_prev - s_prev_prev
            current_min = max(prefix[i], required)
            if current_min > T_total:
                possible = False
                break
            s_prev_prev, s_prev = s_prev, current_min
        if possible and s_prev == T_total:
            print("Yes")
        else:
            print("No")

if __name__ == '__main__':
    main()