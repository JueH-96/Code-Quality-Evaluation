# YOUR CODE HERE
import sys
import threading
def main():
    import bisect
    import math
    N, M, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    total_votes = sum(A)
    R = K - total_votes
    Ai_list = A
    sorted_Ai = sorted(Ai_list, reverse=True)
    prefix_max = sorted_Ai[:M]  # Top M votes
    max_possible_votes = [ai + R for ai in Ai_list]
    sorted_indices = sorted(range(N), key=lambda k: Ai_list[k], reverse=True)
    result = [0]*N
    # For each candidate, compute the minimum number of votes required
    for idx in range(N):
        Ai = Ai_list[idx]
        # Exclude candidate idx from the top M candidates
        temp_top = []
        count = 0
        for i in range(N):
            if i != idx:
                temp_top.append(Ai_list[i] + R)
                count += 1
                if count >= M:
                    break
        temp_top.sort(reverse=True)
        if len(temp_top) < M -1:
            T = -1  # Not enough candidates
        elif len(temp_top) >= M -1:
            T = temp_top[M - 2]
        else:
            T = temp_top[-1]
        Ci = max(0, T +1 - Ai)
        if Ci > R:
            Ci = -1
        result[idx] = Ci
    print(' '.join(map(str, result)))
threading.Thread(target=main).start()