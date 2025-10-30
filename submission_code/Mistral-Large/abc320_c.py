import sys

def min_seconds_to_align_reels(M, S1, S2, S3):
    # Check each possible character from '0' to '9'
    for char in '0123456789':
        positions = []
        # Find all positions of the character in each reel
        for S in [S1, S2, S3]:
            pos = [i for i, x in enumerate(S) if x == char]
            if not pos:
                break
            positions.append(pos)

        if len(positions) == 3:
            # Calculate the minimum time to align the character on all reels
            min_time = float('inf')
            for p1 in positions[0]:
                for p2 in positions[1]:
                    for p3 in positions[2]:
                        max_pos = max(p1, p2, p3)
                        min_time = min(min_time, max_pos + (M - 1) * ((max_pos + M - 1) // M))
            return min_time

    return -1

def main():
    input = sys.stdin.read
    data = input().split()

    M = int(data[0])
    S1 = data[1]
    S2 = data[2]
    S3 = data[3]

    result = min_seconds_to_align_reels(M, S1, S2, S3)
    print(result)

if __name__ == "__main__":
    main()