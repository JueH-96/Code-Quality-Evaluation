# YOUR CODE HERE
from typing import List
from heapq import heappush, heappop

def max_happiness(N: int, M: int, items: List[List[int]]) -> int:
    pull_tabs = []
    cans = []
    can_openers = []
    
    for T, X in items:
        if T == 0:
            pull_tabs.append(-X)
        elif T == 1:
            cans.append(-X)
        else:
            can_openers.append(X)
    
    pull_tabs = sorted(pull_tabs, key=lambda x: abs(x))
    cans = sorted(cans, key=lambda x: abs(x))
    can_openers = sorted(can_openers)
    
    happiness = 0
    can_opener_heap = []
    
    for _ in range(M):
        if not can_opener_heap and not cans and not pull_tabs:
            break
        
        if can_openers and (not can_opener_heap or can_openers[-1] >= -can_opener_heap[0]):
            can_opener_heap.append(can_openers.pop())
        elif cans and (not can_opener_heap or -cans[-1] >= -can_opener_heap[0]):
            happiness -= cans.pop()
        else:
            happiness -= heappop(can_opener_heap)
    
    return -happiness

if __name__ == "__main__":
    N, M = map(int, input().split())
    items = [list(map(int, input().split())) for _ in range(N)]
    print(max_happiness(N, M, items))