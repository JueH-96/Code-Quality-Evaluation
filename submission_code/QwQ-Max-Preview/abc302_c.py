import sys
from itertools import permutations

def main():
    n, m = map(int, sys.stdin.readline().split())
    strings = [sys.stdin.readline().strip() for _ in range(n)]
    
    def is_valid_sequence(seq):
        for i in range(len(seq) - 1):
            a = seq[i]
            b = seq[i+1]
            diff = 0
            for c1, c2 in zip(a, b):
                if c1 != c2:
                    diff += 1
                    if diff > 1:
                        break
            if diff != 1:
                return False
        return True
    
    for perm in permutations(strings):
        if is_valid_sequence(perm):
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()