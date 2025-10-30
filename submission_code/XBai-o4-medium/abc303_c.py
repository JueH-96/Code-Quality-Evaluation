import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr += 1
    M = int(input[ptr]); ptr += 1
    H = int(input[ptr]); ptr += 1
    K = int(input[ptr]); ptr += 1
    S = input[ptr]; ptr += 1

    items = set()
    for _ in range(M):
        x = int(input[ptr])
        y = int(input[ptr + 1])
        items.add((x, y))
        ptr += 2

    x, y = 0, 0
    health = H

    for c in S:
        # Move direction
        if c == 'R':
            x += 1
        elif c == 'L':
            x -= 1
        elif c == 'U':
            y += 1
        elif c == 'D':
            y -= 1

        # Consume health
        health -= 1
        if health < 0:
            print("No")
            return

        # Check for item
        if (x, y) in items:
            if health < K:
                health = K
                items.remove((x, y))

    print("Yes")

if __name__ == "__main__":
    main()