import sys
import itertools

def main():
    N, M = map(int, sys.stdin.readline().split())
    strings = [sys.stdin.readline().strip() for _ in range(N)]
    
    def differ_by_one(s1, s2):
        return sum(a != b for a, b in zip(s1, s2)) == 1
    
    for perm in itertools.permutations(strings):
        valid = True
        for i in range(N-1):
            if not differ_by_one(perm[i], perm[i+1]):
                valid = False
                break
        if valid:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()