import sys
from typing import List

def max_difference(n: int, boxes: List[int]) -> int:
    boxes.sort()
    trucks = [0] * (n // 2)
    for i in range(n // 2):
        trucks[i] = boxes[i] + boxes[n - i - 1]
    trucks.sort()
    return trucks[-1] - trucks[0]

def main():
    t = int(sys.stdin.readline().strip())
    for _ in range(t):
        n = int(sys.stdin.readline().strip())
        boxes = list(map(int, sys.stdin.readline().strip().split()))
        print(max_difference(n, boxes))

if __name__ == "__main__":
    main()