def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = []
    C = []
    for i in range(N):
        S.append(int(data[1 + 2*i]))
        C.append(int(data[2 + 2*i]))
    
    # Create a dictionary to map size to count
    size_count = {}
    for s, c in zip(S, C):
        size_count[s] = c
    
    # Iterate through the sizes in ascending order
    sizes = sorted(size_count.keys())
    for s in sizes:
        c = size_count[s]
        if c < 2:
            continue
        # Calculate the number of new slimes of size 2s
        new_s = 2 * s
        new_c = c // 2
        # Update the count for the new size
        if new_s in size_count:
            size_count[new_s] += new_c
        else:
            size_count[new_s] = new_c
        # Update the count for the current size
        size_count[s] = c % 2
    
    # Calculate the total number of slimes
    total = 0
    for s in size_count:
        total += size_count[s]
    
    print(total)

if __name__ == "__main__":
    main()