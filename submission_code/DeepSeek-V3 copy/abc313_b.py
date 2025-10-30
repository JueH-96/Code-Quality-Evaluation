# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    # Initialize a set to keep track of all programmers
    all_programmers = set(range(1, N+1))
    
    # Initialize a dictionary to store the programmers that are stronger than each programmer
    stronger_than = {i: set() for i in range(1, N+1)}
    
    index = 2
    for _ in range(M):
        A = int(data[index])
        B = int(data[index+1])
        stronger_than[B].add(A)
        index += 2
    
    # Now, for each programmer, we need to find all programmers that are stronger than them
    # We can do this by performing a BFS or DFS to find all reachable nodes
    for programmer in range(1, N+1):
        visited = set()
        stack = [programmer]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                for stronger in stronger_than[current]:
                    if stronger not in visited:
                        stack.append(stronger)
        stronger_than[programmer] = visited - {programmer}
    
    # Now, find the programmer who is stronger than all others
    candidates = []
    for programmer in range(1, N+1):
        # The programmer is stronger than all others if the set of programmers stronger than them is empty
        if not stronger_than[programmer]:
            candidates.append(programmer)
    
    if len(candidates) == 1:
        print(candidates[0])
    else:
        print(-1)

if __name__ == "__main__":
    main()