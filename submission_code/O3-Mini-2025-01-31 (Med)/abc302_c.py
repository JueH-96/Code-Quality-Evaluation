def main():
    import sys, itertools
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    strings = data[2:]
    
    # helper function for calculating the Hamming distance between two strings
    def diff(s, t):
        count = 0
        for a, b in zip(s, t):
            if a != b:
                count += 1
        return count

    # Iterate over all possible orderings of strings.
    for perm in itertools.permutations(strings):
        valid_order = True
        for i in range(N - 1):
            if diff(perm[i], perm[i + 1]) != 1:
                valid_order = False
                break
        if valid_order:
            print("Yes")
            return
    
    print("No")
    
if __name__ == '__main__':
    main()