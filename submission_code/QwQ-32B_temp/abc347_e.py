import sys

def main():
    N, Q = map(int, sys.stdin.readline().split())
    x_list = list(map(int, sys.stdin.readline().split()))

    toggles = [[] for _ in range(N + 1)]
    in_S = [False] * (N + 1)
    current_s = 0
    s = [0] * (Q + 1)  # 1-based indexing

    for i in range(1, Q + 1):
        x = x_list[i - 1]
        if in_S[x]:
            current_s -= 1
            in_S[x] = False
        else:
            current_s += 1
            in_S[x] = True
        s[i] = current_s
        toggles[x].append(i)

    # Compute prefix sums
    prefix = [0] * (Q + 1)
    for i in range(1, Q + 1):
        prefix[i] = prefix[i - 1] + s[i]

    A = [0] * (N + 1)
    for x in range(1, N + 1):
        toggle_list = toggles[x]
        k = len(toggle_list)
        if k == 0:
            A[x] = 0
            continue
        total = 0
        for i in range(0, k, 2):
            a = toggle_list[i]
            if i + 1 < k:
                b = toggle_list[i + 1] - 1
            else:
                b = Q
            total += prefix[b] - prefix[a - 1]
        A[x] = total

    print(' '.join(map(str, A[1:N+1])))

if __name__ == "__main__":
    main()