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
    
    dependencies = collections.defaultdict(list)
    reverse_dependencies = collections.defaultdict(list)
    for i in range(n):
        for prerequisite_book in prerequisites_list[i]:
            dependencies[prerequisite_book].append(i+1)
            reverse_dependencies[i+1].append(prerequisite_book)
            
    required_books_set = set()
    queue = collections.deque([1])
    visited_books = {1}
    
    while queue:
        book_index = queue.popleft()
        for prerequisite_book in reverse_dependencies[book_index]:
            if prerequisite_book not in visited_books:
                visited_books.add(prerequisite_book)
                queue.append(prerequisite_book)
                
    required_books_set = visited_books
    if 1 in required_books_set:
        required_books_set.remove(1)
    else:
        required_books_set = set()
        
    required_books_list = sorted(list(required_books_set))
    if not required_books_list:
        print("")
        return
        
    in_degree = collections.defaultdict(int)
    edges_in_required = []
    for book in required_books_list:
        for prerequisite_book in reverse_dependencies[book]:
            if prerequisite_book in required_books_set:
                edges_in_required.append((prerequisite_book, book))
                
    for u, v in edges_in_required:
        in_degree[v] += 1
        
    sources = []
    for book in required_books_list:
        if in_degree[book] == 0:
            sources.append(book)
            
    sources.sort(reverse=True)
    topological_order = []
    
    while sources:
        current_book = sources.pop(0)
        topological_order.append(current_book)
        
        next_sources = []
        for u, v in edges_in_required:
            if u == current_book:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    next_sources.append(v)
                    
        sources.extend(sorted(next_sources, reverse=True))
        
    print(*(topological_order))

if __name__ == '__main__':
    solve()