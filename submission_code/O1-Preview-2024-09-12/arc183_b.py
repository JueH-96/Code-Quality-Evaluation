# YOUR CODE HERE
import sys
import threading
import bisect

def main():
    import sys
    import bisect
    import math
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    T = int(sys.stdin.readline())
    for _ in range(T):
        N_K_line = ''
        while N_K_line.strip() == '':
            N_K_line = sys.stdin.readline()
        N_str, K_str = N_K_line.strip().split()
        N, K = int(N_str), int(K_str)
        A_line = ''
        while len(A_line.strip().split()) < N:
            A_line += sys.stdin.readline()
        A = list(map(int, A_line.strip().split()))
        B_line = ''
        while len(B_line.strip().split()) < N:
            B_line += sys.stdin.readline()
        B = list(map(int, B_line.strip().split()))
        N = len(A)
        K = int(K)
        A_positions = {}
        B_positions = {}
        for idx, val in enumerate(A):
            if val not in A_positions:
                A_positions[val] = []
            A_positions[val].append(idx + 1)  # positions are 1-based
        for idx, val in enumerate(B):
            if val not in B_positions:
                B_positions[val] = []
            B_positions[val].append(idx + 1)  # positions are 1-based
        possible = True
        for val in B_positions.keys():
            if val not in A_positions:
                possible = False
                break
            A_pos = A_positions[val]
            B_pos = B_positions[val]
            A_pos.sort()
            B_pos.sort()
            for p in B_pos:
                left = bisect.bisect_left(A_pos, p - K)
                if left < len(A_pos) and abs(A_pos[left] - p) <= K:
                    continue
                elif left +1 < len(A_pos) and abs(A_pos[left+1] - p) <= K:
                    continue
                elif left > 0 and abs(A_pos[left -1] - p) <= K:
                    continue
                else:
                    possible = False
                    break
            if not possible:
                break
        if possible:
            print("Yes")
        else:
            print("No")