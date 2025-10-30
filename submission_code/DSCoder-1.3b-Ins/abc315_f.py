import sys
import math

def read_input():
    N = int(sys.stdin.readline().strip())
    points = []
    for _ in range(N):
        x, y = map(int, sys.stdin.readline().strip().split())
        points.append((x, y))
    return N, points

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def solve_problem():
    N, points = read_input()
    total_distance = 0
    for i in range(1, N-1):
        total_distance += calculate_distance(points[i], points[i+1])
    total_distance += calculate_distance(points[0], points[N-1])
    C = N - 1 if N > 0 else 0
    total_distance += 2 ** (C - 1) if C > 0 else 0
    return total_distance

print(solve_problem())