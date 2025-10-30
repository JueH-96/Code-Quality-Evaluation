import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    A = [0] + A  # 1-based indexing

    visited = [False] * (N + 1)

    for i in range(1, N + 1):
        if not visited[i]:
            path = dict()
            path_list = []
            current_node = i
            while True:
                if current_node in path:
                    # Found a cycle
                    start = path[current_node]
                    cycle = path_list[start:]
                    print(len(cycle))
                    print(' '.join(map(str, cycle)))
                    return
                if visited[current_node]:
                    break
                # Add current_node to the current path
                path[current_node] = len(path_list)
                path_list.append(current_node)
                visited[current_node] = True
                current_node = A[current_node]
    # According to problem statement, a solution exists, so this return is just to satisfy syntax
    return

if __name__ == "__main__":
    main()