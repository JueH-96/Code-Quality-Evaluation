def main():
    import sys
    from itertools import permutations
    
    data = sys.stdin.read().split()
    A = list(map(int, data))
    
    for perm in permutations(A):
        if perm[0] * perm[1] == perm[2]:
            print("Yes")
            return
    print("No")

if __name__ == '__main__':
    main()