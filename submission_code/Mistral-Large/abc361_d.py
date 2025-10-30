import sys
from collections import deque

def can_transform(S, T, N):
    # Check if the number of 'W' and 'B' are the same in S and T
    if S.count('W') != T.count('W') or S.count('B') != T.count('B'):
        return -1

    # BFS to find the minimum number of operations
    queue = deque([(S, 0)])
    visited = set([S])

    while queue:
        current, steps = queue.popleft()

        if current == T:
            return steps

        for i in range(1, N):
            if current[i] != T[i] and current[i-1] != T[i-1]:
                new_state = current[:i-1] + current[i-1:i+1][::-1] + current[i+1:]
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, steps + 1))

    return -1

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    S = data[1]
    T = data[2]

    result = can_transform(S, T, N)
    print(result)

if __name__ == "__main__":
    main()