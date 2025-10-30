# YOUR CODE HERE
import sys
import itertools

def differ_by_one(s1, s2):
    diff = 0
    for a, b in zip(s1, s2):
        if a != b:
            diff +=1
        if diff >1:
            return False
    return diff ==1

def main():
    N, M = map(int, sys.stdin.readline().split())
    S_list = [sys.stdin.readline().strip() for _ in range(N)]
    from itertools import permutations
    for perm in itertools.permutations(S_list):
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