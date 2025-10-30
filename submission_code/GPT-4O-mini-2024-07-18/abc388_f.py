def can_move_to_n(N, M, A, B, bad_ranges):
    # Create a list to mark bad squares
    bad = [False] * (N + 1)

    # Mark the bad squares based on the given ranges
    for L, R in bad_ranges:
        for j in range(L, R + 1):
            bad[j] = True

    # Start from square 1
    current_positions = {1}

    # Use a set to track visited positions to avoid cycles
    visited = set()

    while current_positions:
        next_positions = set()
        for pos in current_positions:
            if pos == N:
                return "Yes"
            for step in range(A, B + 1):
                next_pos = pos + step
                if next_pos <= N and not bad[next_pos] and next_pos not in visited:
                    next_positions.add(next_pos)
                    visited.add(next_pos)
        current_positions = next_positions

    return "No"

import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    first_line = data[0].split()
    N = int(first_line[0])
    M = int(first_line[1])
    A = int(first_line[2])
    B = int(first_line[3])
    
    bad_ranges = []
    for i in range(1, M + 1):
        L, R = map(int, data[i].split())
        bad_ranges.append((L, R))
    
    result = can_move_to_n(N, M, A, B, bad_ranges)
    print(result)

if __name__ == "__main__":
    main()