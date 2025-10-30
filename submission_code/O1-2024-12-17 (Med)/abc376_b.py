def main():
    import sys
    from collections import deque

    data = sys.stdin.read().strip().split()
    N, Q = map(int, data[:2])
    # Instructions are pairs of (H_i, T_i)
    instructions = data[2:]

    def ring_distance(n, start, end, blocked):
        """
        Returns the minimum number of steps to move from 'start' to 'end'
        on a ring of size 'n', with one blocked part 'blocked'.
        Movement is 1 <-> 2 <-> 3 <-> ... <-> n <-> 1, with 'blocked' 
        being inaccessible.
        """
        if start == end:
            return 0  # No moves needed

        visited = [False] * (n + 1)
        dist = [-1] * (n + 1)

        visited[start] = True
        dist[start] = 0
        queue = deque([start])

        while queue:
            current = queue.popleft()
            # Two adjacent positions in the ring: next clockwise, next counter-clockwise
            for nxt in [(current % n) + 1, (current - 2 + n) % n + 1]:
                if nxt == blocked:
                    continue
                if not visited[nxt]:
                    visited[nxt] = True
                    dist[nxt] = dist[current] + 1
                    if nxt == end:
                        return dist[nxt]
                    queue.append(nxt)

        # Given the problem guarantees it's always achievable, we shouldn't get here.
        return -1

    # Initial positions of left (L) and right (R) hands
    left_hand, right_hand = 1, 2
    idx = 0
    total_operations = 0

    for _ in range(Q):
        hand = instructions[idx]
        target = int(instructions[idx + 1])
        idx += 2

        if hand == 'L':
            total_operations += ring_distance(N, left_hand, target, right_hand)
            left_hand = target
        else:  # hand == 'R'
            total_operations += ring_distance(N, right_hand, target, left_hand)
            right_hand = target

    print(total_operations)

# Don't forget to call main
if __name__ == "__main__":
    main()