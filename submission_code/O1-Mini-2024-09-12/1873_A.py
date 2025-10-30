# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read().split()
    t = int(input[0])
    for i in range(1, t+1):
        s = input[i]
        target = 'abc'
        if s == target:
            print("YES")
            continue
        mismatches = [j for j in range(3) if s[j] != target[j]]
        if len(mismatches) != 2:
            print("NO")
            continue
        # Try swapping the two mismatched positions
        lst = list(s)
        idx1, idx2 = mismatches
        lst[idx1], lst[idx2] = lst[idx2], lst[idx1]
        if ''.join(lst) == target:
            print("YES")
        else:
            print("NO")

if __name__ == "__main__":
    main()