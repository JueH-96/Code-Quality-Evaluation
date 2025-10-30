# YOUR CODE HERE
import sys
from collections import deque

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    N = int(input[index])
    index += 1
    prerequisites = [[] for _ in range(N + 1)]
    in_degree = [0] * (N + 1)
    
    for i in range(1, N + 1):
        C_i = int(input[index])
        index += 1
        for _ in range(C_i):
            P_i_j = int(input[index])
            index += 1
            prerequisites[P_i_j].append(i)
            in_degree[i] += 1
    
    queue = deque()
    for i in range(1, N + 1):
        if in_degree[i] == 0:
            queue.append(i)
    
    order = []
    while queue:
        current = queue.popleft()
        if current != 1:
            order.append(current)
        for next_book in prerequisites[current]:
            in_degree[next_book] -= 1
            if in_degree[next_book] == 0:
                queue.append(next_book)
    
    print(" ".join(map(str, order)))

if __name__ == "__main__":
    main()