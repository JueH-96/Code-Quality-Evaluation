import sys
from itertools import permutations

def main():
    n, m = map(int, sys.stdin.readline().split())
    words = [sys.stdin.readline().strip() for _ in range(n)]
    
    for perm in permutations(words):
        valid = True
        for i in range(n - 1):
            a = perm[i]
            b = perm[i + 1]
            diff = 0
            for x, y in zip(a, b):
                if x != y:
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

if __name__ == "__main__":
    main()