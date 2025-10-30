import sys

def main():
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1

    queries = []
    for _ in range(Q):
        c = data[ptr]
        k = int(data[ptr+1])
        ptr += 2
        queries.append((c, k))

    # Initialize displacement
    sum_dx = 0
    sum_dy = 0

    for c, k in queries:
        if c == 'U':
            sum_dy += 1
        elif c == 'D':
            sum_dy -= 1
        elif c == 'L':
            sum_dx -= 1
        elif c == 'R':
            sum_dx += 1

        # For each query, compute the position
        for _ in range(k):
            # Compute displacement for each part
            # But since all parts are affected, we can compute for each part
            # However, for large N, this is O(N) per query which is too slow.
            # Instead, we need a different approach.
            # The correct position for part p is:
            # x = head.x + (sum_dx) * (p-1)
            # y = head.y + (sum_dy) * (p-1)
            # But wait, after each movement, the head's position changes.
            # So, we need to track the head's position.
            # So, we need to track head.x and head.y after each movement.

            # So, we should track head's position as well.
            # Let's adjust the approach.

            # Track head's position after each movement.
            # Initially, head is at (1, 0)
            head_x = 1
            head_y = 0

            # For each movement, update head's position
            if c == 'U':
                head_y += 1
            elif c == 'D':
                head_y -= 1
            elif c == 'L':
                head_x -= 1
            elif c == 'R':
                head_x += 1

            # Now, for the current query, after k movements, compute the position for each part in the next k queries.
            # But for each query, the number of parts to compute is up to N, which is 2e5 per query.
            # This is too slow. So, we need a way to compute the position of a specific part quickly.

            # The correct position for part p is:
            # x = head_x - (p - 1) + sum_dx * (p - 1)
            # y = head_y + sum_dy * (p - 1)
            # Wait, let's re-express this.

            # The initial position of part p is (p, 0)
            # Each movement changes the head's position, and each part p's position is determined by the movement of the head and the previous positions of the parts.

            # After K movements, the position of part p is:
            # x = head_x - (p - 1) + sum_dx * (p - 1)
            # y = head_y + sum_dy * (p - 1)

            # Let's test this with the sample.

            # Sample after first movement: R
            # movement R: head_x = 2, head_y =0
            # sum_dx=1, sum_dy=0
            # For p=3:
            # x = 2 - 2 + 1*2 = 0 + 2 = 2
            # y = 0 + 0*2 = 0
            # Which matches the sample.

            # So the correct formula is:
            x = head_x - (p - 1) + sum_dx * (p - 1)
            y = head_y + sum_dy * (p - 1)

            print(x, y)

if __name__ == '__main__':
    main()