import collections

def solve():
    n = int(input())
    prerequisites_list = []
    for _ in range(n):
        line = list(map(int, input().split()))
        count = line[0]
        if count > 0:
            prerequisites = line[1:]
        else:
            prerequisites = []
        prerequisites_list.append(prerequisites)
    
    original_dependencies = collections.defaultdict(list)
    reversed_dependencies = collections.defaultdict(list)
    for i in range(n):
        book_index = i + 1
        for prerequisite_book in prerequisites_list[i]:
            original_dependencies[prerequisite_book].append(book_index)
            reversed_dependencies[book_index].append(prerequisite_book)
            
    required_books = set()
    queue = collections.deque([1])
    visited = {1}
    while queue:
        book = queue.popleft()
        for prerequisite in reversed_dependencies[book]:
            if prerequisite not in visited:
                visited.add(prerequisite)
                queue.append(prerequisite)
                
    required_books = visited
    if 1 in required_books:
        required_books.remove(1)
    else:
        required_books = set()
        
    required_books_list = sorted(list(required_books))
    if not required_books_list:
        print("")
        return
        
    required_dependencies = collections.defaultdict(list)
    in_degree = collections.defaultdict(int)
    
    for book in required_books_list:
        for prerequisite in prerequisites_list[book-1]:
            if prerequisite in required_books:
                required_dependencies[prerequisite].append(book)
                in_degree[book] += 1
                
    zero_in_degree_queue = collections.deque()
    for book in required_books_list:
        if in_degree[book] == 0:
            zero_in_degree_queue.append(book)
            
    reading_order = []
    while zero_in_degree_queue:
        current_book = zero_in_degree_queue.popleft()
        reading_order.append(current_book)
        for dependent_book in required_dependencies[current_book]:
            in_degree[dependent_book] -= 1
            if in_degree[dependent_book] == 0:
                zero_in_degree_queue.append(dependent_book)
                
    print(*(reading_order))

if __name__ == '__main__':
    solve()