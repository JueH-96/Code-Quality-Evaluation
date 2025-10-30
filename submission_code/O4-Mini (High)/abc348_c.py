import sys
import threading

def main():
    import sys

    input = sys.stdin.readline
    N = int(input())
    # Map from color to current minimum deliciousness for that color
    min_del_per_color = {}
    for _ in range(N):
        A, C = map(int, input().split())
        if C in min_del_per_color:
            if A < min_del_per_color[C]:
                min_del_per_color[C] = A
        else:
            min_del_per_color[C] = A

    # Among all colors, take the maximum of these minima
    answer = max(min_del_per_color.values())
    print(answer)

if __name__ == "__main__":
    main()