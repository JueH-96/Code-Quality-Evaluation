import sys

def main():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    total_sum = sum(A)

    # Binary search for the maximum x
    low = 1
    high = total_sum
    best_x = 0

    def is_possible(L):
        if total_sum < L * K:
            return False
        current_sum = 0
        groups = 0
        prev = 0
        can_merge = False
        for a in A:
            current_sum += a
            if current_sum >= L:
                groups += 1
                if prev + current_sum >= L:
                    can_merge = True
                prev = current_sum
                current_sum = 0
        if current_sum > 0:
            groups += 1
            if prev + current_sum >= L:
                can_merge = True
        if groups <= K:
            return True
        else:
            if can_merge and (groups - 1) <= K:
                return True
            else:
                return False

    while low <= high:
        mid = (low + high) // 2
        if is_possible(mid):
            best_x = mid
            low = mid + 1
        else:
            high = mid - 1

    x = best_x

    # Compute prefix_groups and prefix_remaining
    prefix_groups = [0] * (N + 1)
    prefix_remaining = [0] * (N + 1)
    current_sum = 0
    groups = 0
    for i in range(1, N + 1):
        current_sum += A[i - 1]
        if current_sum >= x:
            groups += 1
            current_sum = 0
        prefix_groups[i] = groups
        prefix_remaining[i] = current_sum

    # Compute suffix_groups and suffix_remaining
    suffix_groups = [0] * (N + 2)  # suffix_groups[i] is for suffix starting at i
    suffix_remaining = [0] * (N + 2)
    current_sum = 0
    groups = 0
    for i in range(N, 0, -1):
        current_sum += A[i - 1]
        if current_sum >= x:
            groups += 1
            current_sum = 0
        suffix_groups[i] = groups
        suffix_remaining[i] = current_sum

    count = 0
    for i in range(1, N + 1):
        # First part: 1..i
        groups_first = prefix_groups[i]
        rem_first = prefix_remaining[i]
        groups_first += 1 if rem_first > 0 else 0

        # Second part: i+1..N
        groups_second = suffix_groups[i + 1]
        rem_second = suffix_remaining[i + 1]
        groups_second += 1 if rem_second > 0 else 0

        if (groups_first + groups_second) <= K and groups_first >= 1 and groups_second >= 1:
            continue
        else:
            count += 1

    print(best_x, count)

if __name__ == "__main__":
    main()