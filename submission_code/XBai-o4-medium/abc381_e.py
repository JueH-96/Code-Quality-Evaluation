import bisect
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    S = input[ptr]
    ptr += 1

    # Compute prefix sums for '1's and '2's
    prefix_1 = [0] * (N + 1)
    prefix_2 = [0] * (N + 1)
    for i in range(1, N + 1):
        c = S[i - 1]
        prefix_1[i] = prefix_1[i - 1] + (1 if c == '1' else 0)
        prefix_2[i] = prefix_2[i - 1] + (1 if c == '2' else 0)

    # Collect positions of '/' (1-based)
    sl = []
    for i in range(N):
        if S[i] == '/':
            sl.append(i + 1)  # Convert to 1-based index

    # Process each query
    for _ in range(Q):
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1

        # Find the indices of slashes in [L, R]
        left_idx = bisect.bisect_left(sl, L)
        right_idx = bisect.bisect_right(sl, R) - 1

        if left_idx > right_idx or right_idx < 0:
            print(0)
            continue

        low = left_idx
        high = right_idx
        best_val = 0

        while low <= high:
            if high - low <= 3:
                # Check all positions from low to high
                for i in range(low, high + 1):
                    pos = sl[i]
                    a = prefix_1[pos - 1] - prefix_1[L - 1]
                    b = prefix_2[R] - prefix_2[pos]
                    current = min(a, b)
                    if current > best_val:
                        best_val = current
                break
            else:
                m1 = low + (high - low) // 3
                m2 = high - (high - low) // 3
                # Evaluate m1
                pos1 = sl[m1]
                a1 = prefix_1[pos1 - 1] - prefix_1[L - 1]
                b1 = prefix_2[R] - prefix_2[pos1]
                val1 = min(a1, b1)
                # Evaluate m2
                pos2 = sl[m2]
                a2 = prefix_1[pos2 - 1] - prefix_1[L - 1]
                b2 = prefix_2[R] - prefix_2[pos2]
                val2 = min(a2, b2)
                if val1 < val2:
                    low = m1 + 1
                else:
                    high = m2 - 1

        ans = 2 * best_val + 1
        print(ans)

if __name__ == "__main__":
    main()