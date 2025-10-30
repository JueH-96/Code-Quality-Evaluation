import sys

def main() -> None:
    # Read N and K
    N_K_line = sys.stdin.readline().strip()
    while N_K_line == '':
        N_K_line = sys.stdin.readline().strip()
    N, K = map(int, N_K_line.split())

    # Read the sequence A (may span multiple lines)
    nums_needed = N
    A_vals = []
    while nums_needed > 0:
        line = sys.stdin.readline()
        if not line:
            break
        parts = line.strip().split()
        if not parts:
            continue
        A_vals.extend(map(int, parts))
        nums_needed -= len(parts)

    # Keep only unique values in [1, K]
    seen = set()
    for a in A_vals:
        if 1 <= a <= K:
            seen.add(a)

    # Sum of 1..K
    total_sum = K * (K + 1) // 2
    # Subtract the sum of present numbers
    missing_sum = total_sum - sum(seen)

    print(missing_sum)


if __name__ == "__main__":
    main()