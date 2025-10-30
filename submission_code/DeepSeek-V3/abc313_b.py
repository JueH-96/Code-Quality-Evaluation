# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    # Initialize a set to keep track of all programmers
    all_programmers = set(range(1, N+1))
    
    # Initialize a dictionary to keep track of who is stronger than whom
    stronger = {i: set() for i in range(1, N+1)}
    
    index = 2
    for _ in range(M):
        A = int(data[index])
        B = int(data[index+1])
        stronger[A].add(B)
        index += 2
    
    # Transitive closure to fill in all implied relationships
    for k in range(1, N+1):
        for i in range(1, N+1):
            if k in stronger[i]:
                for j in range(1, N+1):
                    if j in stronger[k]:
                        stronger[i].add(j)
    
    # Now, find the candidate(s) who are stronger than all others
    candidates = []
    for x in range(1, N+1):
        is_candidate = True
        for y in range(1, N+1):
            if x == y:
                continue
            if y not in stronger[x]:
                is_candidate = False
                break
        if is_candidate:
            candidates.append(x)
    
    if len(candidates) == 1:
        print(candidates[0])
    else:
        print(-1)

if __name__ == "__main__":
    main()