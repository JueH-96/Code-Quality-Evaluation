import sys
from typing import List

def min_operations(n: int, a: List[int]) -> int:
    a.sort()
    return min(a[i+1] - a[i] for i in range(n - 1))

def main():
    n = int(sys.stdin.readline().strip())
    a = list(map(int, sys.stdin.readline().strip().split()))
    print(min_operations(n, a))

if __name__ == "__main__":
    main()