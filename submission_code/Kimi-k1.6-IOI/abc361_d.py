import sys
from collections import deque

def main():
    n = int(sys.stdin.readline())
    s = sys.stdin.readline().strip()
    t = sys.stdin.readline().strip()
    
    if sorted(s) != sorted(t):
        print(-1)
        return
    
    initial_stones = s + '..'
    target_stones = t + '..'
    initial_k = n  # 0-based index for the first empty cell (empty cells are at n and n+1)
    
    visited = set()
    queue = deque()
    visited.add((initial_k, initial_stones))
    queue.append((initial_k, initial_stones, 0))
    
    found = False
    answer = -1
    
    while queue:
        k, stones, steps = queue.popleft()
        
        # Check if current state matches the target
        if stones[:n] == t and k == n:
            answer = steps
            found = True
            break
        
        # Generate all possible moves
        for x in range(len(stones) - 1):
            if stones[x] != '.' and stones[x+1] != '.':
                # Create new stones configuration
                new_stones = list(stones)
                new_stones[k] = new_stones[x]
                new_stones[k+1] = new_stones[x+1]
                new_stones[x] = '.'
                new_stones[x+1] = '.'
                new_stones_str = ''.join(new_stones)
                new_k = x
                # Check if this new state has been visited
                if (new_k, new_stones_str) not in visited:
                    visited.add((new_k, new_stones_str))
                    queue.append((new_k, new_stones_str, steps + 1))
    
    print(answer if found else -1)

if __name__ == "__main__":
    main()