import sys
from typing import List, Tuple

def solve(n: int, m: int, k: int, a: List[int]) -> List[int]:
    total_votes = sum(a)
    remaining_votes = k - total_votes
    if remaining_votes < 0:
        raise ValueError("Not enough votes")
    a.sort()
    result = []
    for i in range(n):
        if a[i] + remaining_votes >= a[-m]:
            result.append(0)
        else:
            low, high = 0, remaining_votes + 1
            while high - low > 1:
                mid = (low + high) // 2
                if a[i] + mid >= a[-m]:
                    high = mid
                else:
                    low = mid
            result.append(high)
    return result

def main():
    n, m, k = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    try:
        print(*solve(n, m, k, a))
    except ValueError:
        print("Not enough votes")

if __name__ == "__main__":
    main()