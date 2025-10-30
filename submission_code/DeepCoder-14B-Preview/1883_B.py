import sys
from collections import Counter

def can_rearrange_palindrome(n, k, s):
    m = n - k
    if m <= 0:
        return False
    cnt = Counter(s)
    O = sum(1 for v in cnt.values() if v % 2 != 0)
    required_O = m % 2
    delta = abs(O - required_O)
    if k >= delta and (k - delta) % 2 == 0:
        return True
    return False

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n, k = map(int, sys.stdin.readline().split())
        s = sys.stdin.readline().strip()
        if can_rearrange_palindrome(n, k, s):
            print("YES")
        else:
            print("NO")

if __name__ == "__main__":
    main()