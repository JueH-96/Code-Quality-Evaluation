def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    T = int(input[idx+1])
    M = int(input[idx+2])
    idx +=3
    incompatible = []
    for _ in range(M):
        a = int(input[idx])-1
        b = int(input[idx+1])-1
        incompatible.append( (a, b) )
        idx +=2
    
    count = 0
    
    def backtrack(assignment, current_max, index):
        nonlocal count
        if index == N:
            if current_max == T-1:
                # Check all incompatible pairs
                valid = True
                for a, b in incompatible:
                    if assignment[a] == assignment[b]:
                        valid = False
                        break
                if valid:
                    count +=1
            return
        # Assign to existing subsets
        for s in range(current_max + 1):
            backtrack(assignment + [s], current_max, index + 1)
        # Assign to new subset if possible
        if current_max + 1 < T:
            backtrack(assignment + [current_max + 1], current_max + 1, index + 1)
    
    if T ==0 or N < T:
        print(0)
        return
    
    backtrack([], -1, 0)
    print(count)

if __name__ == "__main__":
    main()