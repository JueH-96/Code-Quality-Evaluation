import sys
import math

def main() -> None:
    # Read all input at once for speed
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n = int(data[0])

    # We will walk from (0,0) through all points and back to (0,0)
    px, py = 0, 0           # previous (current) position
    total = 0.0             # accumulated distance
    idx = 1                 # index in the token list

    for _ in range(n):
        x = int(data[idx]); y = int(data[idx + 1])
        idx += 2
        total += math.hypot(x - px, y - py)   # distance to the next point
        px, py = x, y                         # update current position

    # finally return to the origin
    total += math.hypot(px, py)

    # Print with enough digits to guarantee 1e-6 accuracy
    print("{:.15f}".format(total))

if __name__ == "__main__":
    main()