# YOUR CODE HERE
from sys import stdin
import threading
def main():
    import sys

    import bisect
    N = int(stdin.readline())
    A = list(map(int, stdin.readline().split()))
    M = int(stdin.readline())
    B = list(map(int, stdin.readline().split()))
    L = int(stdin.readline())
    C = list(map(int, stdin.readline().split()))
    Q = int(stdin.readline())
    X = list(map(int, stdin.readline().split()))

    sum_ab = set()
    for a in A:
        for b in B:
            sum_ab.add(a+b)

    sum_ab = set(sum_ab)
    answers = []

    for xi in X:
        found = False
        for c in C:
            if xi - c in sum_ab:
                found = True
                break
        if found:
            print("Yes")
        else:
            print("No")
threading.Thread(target=main).start()