import sys

def main():
    N, Q = map(int, sys.stdin.readline().split())
    parts = [[i, 0] for i in range(1, N + 1)]

    for _ in range(Q):
        query = sys.stdin.readline().split()
        if query[0] == '1':
            direction = query[1]
            if direction == 'R':
                parts[0][0] += 1
            elif direction == 'L':
                parts[0][0] -= 1
            elif direction == 'U':
                parts[0][1] += 1
            elif direction == 'D':
                parts[0][1] -= 1
            for i in range(1, N):
                parts[i] = parts[i - 1].copy()
        elif query[0] == '2':
            p = int(query[1])
            print(parts[p - 1][0], parts[p - 1][1])

if __name__ == "__main__":
    main()