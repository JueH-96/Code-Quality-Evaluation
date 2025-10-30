import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    D = int(input[ptr])
    ptr += 1

    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+M]))
    ptr += M

    A_sorted = sorted(A, reverse=True)
    B_sorted = sorted(B)

    max_sum = -1

    for a in A_sorted:
        high = a + D
        idx = bisect.bisect_right(B_sorted, high) - 1
        if idx >= 0:
            b_candidate = B_sorted[idx]
            if b_candidate >= a - D:
                current_sum = a + b_candidate
                if current_sum > max_sum:
                    max_sum = current_sum
                # Check for early termination
                if (a * 2 + D) <= max_sum:
                    break

    print(max_sum if max_sum != -1 else -1)

if __name__ == '__main__':
    main()