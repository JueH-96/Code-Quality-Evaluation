def main():
    import sys
    import heapq

    # Read input
    N, M, L = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    b = list(map(int, sys.stdin.readline().split()))
    forbidden = set()
    for _ in range(L):
        c, d = map(int, sys.stdin.readline().split())
        forbidden.add((c-1, d-1))  # zero-indexed

    # Sort a and b in descending order, keeping track of original indices
    a_sorted = sorted([(a[i], i) for i in range(N)], reverse=True)
    b_sorted = sorted([(b[j], j) for j in range(M)], reverse=True)

    # Choose k such that k^2 > L, e.g., k=317
    k = 317
    a_top = a_sorted[:k]
    b_top = b_sorted[:k]

    # Find the maximum sum among allowed pairs
    max_sum = -1
    for ai, ai_idx in a_top:
        for bj, bj_idx in b_top:
            if (ai_idx, bj_idx) not in forbidden:
                max_sum = max(max_sum, ai + bj)
                # Since lists are sorted descending, can break early
                break
        if max_sum != -1:
            break
    # If not found in the first a_top, iterate through all a_top and b_top
    if max_sum == -1:
        for ai, ai_idx in a_top:
            for bj, bj_idx in b_top:
                if (ai_idx, bj_idx) not in forbidden:
                    max_sum = max(max_sum, ai + bj)
                    if max_sum == a_top[0][0] + b_top[0][0]:
                        break  # Can't get higher than this
            if max_sum == a_top[0][0] + b_top[0][0]:
                break

    print(max_sum)

if __name__ == "__main__":
    main()