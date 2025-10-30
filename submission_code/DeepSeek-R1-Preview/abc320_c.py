import sys
from collections import defaultdict

def main():
    M = int(sys.stdin.readline())
    S1 = sys.stdin.readline().strip()
    S2 = sys.stdin.readline().strip()
    S3 = sys.stdin.readline().strip()

    # Preprocess each reel to map each digit to its positions
    reels = [S1, S2, S3]
    positions = [defaultdict(list) for _ in range(3)]
    for i in range(3):
        for pos, char in enumerate(reels[i]):
            positions[i][char].append(pos)

    min_time = float('inf')

    # Check each possible digit
    for d in map(str, range(10)):
        # Check if the digit exists in all three reels
        if d not in positions[0] or d not in positions[1] or d not in positions[2]:
            continue

        # Get the list of positions for each reel
        p1_list = positions[0][d]
        p2_list = positions[1][d]
        p3_list = positions[2][d]

        # Iterate all possible triplets
        for p1 in p1_list:
            for p2 in p2_list:
                for p3 in p3_list:
                    x, y, z = sorted([p1, p2, p3])

                    # Compute candidate time
                    if x != y and y != z:
                        candidate = z
                    else:
                        if x == y == z:
                            candidate = x + 2 * M
                        elif x == y:
                            candidate = max(z, x + M)
                        elif y == z:
                            candidate = y + M
                        else:
                            # This case shouldn't happen as per sorted and previous checks
                            candidate = float('inf')

                    if candidate < min_time:
                        min_time = candidate

    if min_time == float('inf'):
        print(-1)
    else:
        print(min_time)

if __name__ == '__main__':
    main()