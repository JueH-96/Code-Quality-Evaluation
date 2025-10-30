import sys
from collections import deque, defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    index = 1
    prerequisites = defaultdict(list)
    in_degree = [0] * (N + 1)
    
    for i in range(1, N + 1):
        C = int(data[index])
        index += 1
        for _ in range(C):
            P = int(data[index])
            prerequisites[P].append(i)
            in_degree[i] += 1
            index += 1
    
    queue = deque()
    for i in range(1, N + 1):
        if in_degree[i] == 0:
            queue.append(i)
    
    order = []
    while queue:
        book = queue.popleft()
        order.append(book)
        for next_book in prerequisites[book]:
            in_degree[next_book] -= 1
            if in_degree[next_book] == 0:
                queue.append(next_book)
    
    required_books = []
    for book in reversed(order):
        if book in prerequisites[1]:
            required_books.append(book)
    
    required_books.reverse()
    print(" ".join(map(str, required_books)))

if __name__ == "__main__":
    main()