def main():
    import sys
    from itertools import permutations
    
    data = sys.stdin.read().split()
    if not data:
        return
    
    N = int(data[0])
    M = int(data[1])
    strings = data[2:]
    
    # For each permutation of strings, check if for every consecutive pair
    # the hamming distance (number of different characters between the two
    # strings) is exactly 1.
    for perm in permutations(strings, N):
        valid = True
        for i in range(N - 1):
            # Count the number of differing characters between perm[i] and perm[i+1]
            diff = 0
            for a, b in zip(perm[i], perm[i+1]):
                if a != b:
                    diff += 1
                if diff > 1:
                    break
            if diff != 1:
                valid = False
                break
        if valid:
            print("Yes")
            return
            
    print("No")

if __name__ == '__main__':
    main()