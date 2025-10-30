import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    T = input[idx + 1]
    idx += 2
    S_list = input[idx:idx + N]
    L = len(T)
    if L == 0:
        print(0)
        return

    left_freq = [0] * (L + 1)
    cnt = [0] * (L + 2)  # cnt[y] counts right_max y in 0..L

    for S in S_list:
        # Compute left_max
        t_ptr = 0
        len_t = L
        for c in S:
            if t_ptr < len_t and T[t_ptr] == c:
                t_ptr += 1
        left = t_ptr
        left_freq[left] += 1

        # Compute right_max
        t_ptr = L - 1
        for c in reversed(S):
            if t_ptr >= 0 and T[t_ptr] == c:
                t_ptr -= 1
        right = (L - 1) - t_ptr
        cnt[right] += 1

    # Build suffix sum array
    suffix = [0] * (L + 2)
    suffix[L + 1] = 0
    for y in range(L, -1, -1):
        suffix[y] = cnt[y] + suffix[y + 1]

    # Calculate total valid pairs
    total = 0
    for x in range(L + 1):
        if left_freq[x] == 0:
            continue
        y_min = max(0, L - x)
        if y_min > L:
            sum_right = 0
        else:
            sum_right = suffix[y_min]
        total += left_freq[x] * sum_right

    print(total)

if __name__ == "__main__":
    main()