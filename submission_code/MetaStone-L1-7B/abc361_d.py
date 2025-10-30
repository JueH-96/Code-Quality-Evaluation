from collections import deque

def main():
    import sys
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    if S == T:
        print(0)
        return

    initial_stones = {i for i in range(1, N+1) if S[i-1] == 'B'}
    target_stones = {i for i in range(1, N+1) if T[i-1] == 'B'}

    if initial_stones == target_stones and (N+1) == (N+1):
        print(0)
        return

    visited = set()
    queue = deque()
    a_initial = N + 1
    initial_state = (a_initial, frozenset(initial_stones))
    queue.append(initial_state)
    visited.add(initial_state)

    found = False
    answer = -1

    while queue:
        a, stones = queue.popleft()

        if a == N + 1 and stones == target_stones:
            answer = len(queue)  # The number of steps is the current length of the queue
            found = True
            break

        for x in list(stones):
            if (x + 1) in stones:
                new_stones = set(stones)
                new_stones.discard(x)
                new_stones.discard(x + 1)
                new_stones.add(a)
                new_stones.add(a + 1)
                new_stones_frozen = frozenset(new_stones)
                new_a = x
                new_state = (new_a, new_stones_frozen)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append(new_state)

    if found:
        print(answer)
    else:
        print(-1)

if __name__ == '__main__':
    main()