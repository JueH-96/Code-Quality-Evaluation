import sys
import bisect

def main():
    import sys
    import sys
    from bisect import bisect_left

    input = sys.stdin.read().split('
')
    first_line = input[0].split()
    N = int(first_line[0])
    T = first_line[1]
    S_list = input[1:N+1]

    T_len = len(T)
    T_reversed = T[::-1]

    # Precompute f_i for all S_i
    f_list = []
    for S in S_list:
        j = 0
        for ch in S:
            if j < T_len and ch == T[j]:
                j +=1
        f_list.append(j)

    # Precompute g_j for all S_j
    g_list = []
    for S in S_list:
        S_rev = S[::-1]
        j = 0
        for ch in S_rev:
            if j < T_len and ch == T_reversed[j]:
                j +=1
        g_list.append(j)

    # Sort g_list
    g_sorted = sorted(g_list)

    result = 0
    for f in f_list:
        x = T_len - f
        if x <=0:
            # All g_j satisfy
            result += N
        elif x > T_len:
            # No g_j satisfy
            continue
        else:
            pos = bisect_left(g_sorted, x)
            result += N - pos

    print(result)

if __name__ == "__main__":
    main()