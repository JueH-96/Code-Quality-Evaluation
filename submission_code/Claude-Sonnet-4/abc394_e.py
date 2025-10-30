from collections import deque

def solve():
    n = int(input())
    graph = []
    for i in range(n):
        row = input().strip()
        graph.append(row)
    
    result = []
    
    for start in range(n):
        row_result = []
        for end in range(n):
            # BFS with state (vertex, stack_of_unmatched_chars)
            # The stack represents characters that need to be matched in reverse order
            queue = deque([(start, tuple(), 0)])  # (vertex, unmatched_stack, length)
            visited = set()
            visited.add((start, tuple()))
            
            min_length = -1
            
            while queue:
                curr_vertex, stack, length = queue.popleft()
                
                # If we're at target vertex with empty stack, we have a palindrome
                if curr_vertex == end and len(stack) == 0:
                    min_length = length
                    break
                
                # Try all outgoing edges
                for next_vertex in range(n):
                    if graph[curr_vertex][next_vertex] != '-':
                        edge_char = graph[curr_vertex][next_vertex]
                        
                        # Two cases:
                        # 1. If stack is not empty and top matches current char, pop it
                        # 2. Otherwise, push current char onto stack
                        
                        if stack and stack[-1] == edge_char:
                            # Match found, pop from stack
                            new_stack = stack[:-1]
                        else:
                            # No match, push to stack
                            new_stack = stack + (edge_char,)
                        
                        state = (next_vertex, new_stack)
                        if state not in visited:
                            visited.add(state)
                            queue.append((next_vertex, new_stack, length + 1))
            
            row_result.append(min_length)
        result.append(row_result)
    
    for row in result:
        print(*row)

solve()