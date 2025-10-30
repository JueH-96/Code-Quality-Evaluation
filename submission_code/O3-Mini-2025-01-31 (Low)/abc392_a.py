def main():
    import sys
    from itertools import permutations
    data = sys.stdin.read().split()
    
    # Convert data to integers
    A = list(map(int, data))
    # There should be exactly 3 numbers
    if len(A) != 3:
        return  # In case there is some error in input

    possible = False
    # Check over all permutations
    for perm in permutations(A):
        if perm[0] * perm[1] == perm[2]:
            possible = True
            break

    if possible:
        print("Yes")
    else:
        print("No")
        
if __name__ == '__main__':
    main()