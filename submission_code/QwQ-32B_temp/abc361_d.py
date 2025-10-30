import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # Check if S and T have the same multiset of characters
    if sorted(S) != sorted(T):
        print(-1)
        return

    initial = S + '..'
    target = T + '..'

    if initial == target:
        print(0)
        return

    visited = {}
    q = deque()
    q.append(initial)
    visited[initial] = 0

    while q:
        current = q.popleft()
        if current == target:
            print(visited[current])
            return

        # Find the positions of the empty cells (pair of '.')
        k = -1
        for i in range(len(current) - 1):
            if current[i] == '.' and current[i+1] == '.':
                k = i
                break

        # Iterate over all possible pairs of adjacent stones
        for x in range(len(current) - 1):
            if current[x] == '.' or current[x+1] == '.':
                continue
            # Check if the pair is not overlapping with empty positions
            if (x + 1 < k) or (x > k + 1):
                # Create new configuration
                new_config = list(current)
                # Move the stones from x and x+1 to k and k+1
                new_config[x] = '.'
                new_config[x+1] = '.'
                new_config[k] = current[x]
                new_config[k+1] = current[x+1]
                new_str = ''.join(new_config)
                if new_str not in visited:
                    visited[new_str] = visited[current] + 1
                    q.append(new_str)

    # If target is unreachable
    print(-1)

if __name__ == '__main__':
    main()