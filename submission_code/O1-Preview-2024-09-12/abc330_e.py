# YOUR CODE HERE
import sys
import threading
import heapq
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    cnt_size = N + 2  # to include N+1
    cnt = [0] * (cnt_size)
    min_mex_heap = []
    for i in range(cnt_size):
        heapq.heappush(min_mex_heap, i)
    for a in A:
        if a < cnt_size:
            cnt[a] +=1
    # Remove initial elements from heap that are already in A
    while cnt[min_mex_heap[0]] > 0:
        heapq.heappop(min_mex_heap)

    for _ in range(Q):
        s = ''
        while s.strip() == '':
            s = sys.stdin.readline()
        i_k_str, x_k_str = s.strip().split()
        i_k = int(i_k_str) -1   # zero-based index
        x_k = int(x_k_str)
        old_value = A[i_k]
        if old_value < cnt_size:
            cnt[old_value] -=1
            if cnt[old_value]==0:
                heapq.heappush(min_mex_heap, old_value)
        A[i_k] = x_k
        if x_k < cnt_size:
            cnt[x_k] +=1
        # Adjust mex
        while True:
            mex_candidate = min_mex_heap[0]
            if cnt[mex_candidate]==0:
                print(mex_candidate)
                break
            else:
                heapq.heappop(min_mex_heap)

threading.Thread(target=main).start()