import sys
import math
from typing import List, Tuple

def earliest_arrival_time(N: int, X: int, Y: int, bus_info: List[Tuple[int, int]], Q: int, queries: List[int]) -> List[int]:
    bus_info = [(X,)] + bus_info + [(Y,)]
    for _ in range(Q):
        q = queries[_]
        t = q
        for i in range(N):
            P, T = bus_info[i]
            t = ((t + P - 1) // P) * P + T
        print(t)

def main():
    N, X, Y = map(int, input().split())
    bus_info = [tuple(map(int, input().split())) for _ in range(N - 1)]
    Q = int(input())
    queries = [int(input()) for _ in range(Q)]
    earliest_arrival_time(N, X, Y, bus_info, Q, queries)

if __name__ == "__main__":
    main()