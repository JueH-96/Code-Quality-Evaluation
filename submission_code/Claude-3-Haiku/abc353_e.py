from typing import List

def longest_common_prefix(x: str, y: str) -> int:
    i = 0
    while i < min(len(x), len(y)) and x[i] == y[i]:
        i += 1
    return i

def solve(strings: List[str]) -> int:
    n = len(strings)
    total = 0
    for i in range(n-1):
        for j in range(i+1, n):
            total += longest_common_prefix(strings[i], strings[j])
    return total

if __name__ == "__main__":
    n = int(input())
    strings = [input() for _ in range(n)]
    print(solve(strings))