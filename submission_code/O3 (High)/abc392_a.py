import sys
import itertools

def main():
    # Read the three integers
    nums = list(map(int, sys.stdin.read().strip().split()))
    
    # Check every permutation to see if the first two multiply to the third
    for p in itertools.permutations(nums):
        if p[0] * p[1] == p[2]:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()