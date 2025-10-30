from collections import deque

def main():
    import sys
    K = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    len_S = len(S)
    len_T = len(T)

    queue = deque()
    queue.append((0, 0, 0))
    visited = dict()

    while queue:
        i, j, ops = queue.popleft()

        if i == len_S and j == len_T:
            print("Yes")
            return

        if ops > K:
            continue

        # Operation 1: delete S[i]
        if i + 1 <= len_S:
            new_i = i + 1
            new_j = j
            new_ops = ops + 1
            if (new_i, new_j) not in visited or new_ops < visited.get((new_i, new_j), float('inf')):
                visited[(new_i, new_j)] = new_ops
                queue.append((new_i, new_j, new_ops))

        # Operation 2: delete T[j]
        if j + 1 <= len_T:
            new_i = i
            new_j = j + 1
            new_ops = ops + 1
            if (new_i, new_j) not in visited or new_ops < visited.get((new_i, new_j), float('inf')):
                visited[(new_i, new_j)] = new_ops
                queue.append((new_i, new_j, new_ops))

        # Operation 3: replace S[i] with T[j]
        if i <= len_S and j <= len_T:
            new_i = i + 1
            new_j = j + 1
            new_ops = ops + 1
            if new_i <= len_S and new_j <= len_T:
                if (new_i, new_j) not in visited or new_ops < visited.get((new_i, new_j), float('inf')):
                    visited[(new_i, new_j)] = new_ops
                    queue.append((new_i, new_j, new_ops))

        # Operation 4: insert T[j] into S
        if j + 1 <= len_T and i <= len_S:
            new_i = i
            new_j = j + 1
            new_ops = ops + 1
            if (new_i, new_j) not in visited or new_ops < visited.get((new_i, new_j), float('inf')):
                visited[(new_i, new_j)] = new_ops
                queue.append((new_i, new_j, new_ops))

    print("No")

if __name__ == "__main__":
    main()