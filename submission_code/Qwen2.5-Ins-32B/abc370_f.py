import sys

def can_divide(min_weight, pieces, K):
    total, count = 0, 1
    for piece in pieces:
        if total + piece >= min_weight:
            count += 1
            total = 0
        else:
            total += piece
    return count >= K + 1

def find_min_weight(pieces, K):
    low, high = max(pieces), sum(pieces)
    while low < high:
        mid = (low + high + 1) // 2
        if can_divide(mid, pieces, K):
            low = mid
        else:
            high = mid - 1
    return low

def count_unchanged_cuts(pieces, min_weight, K):
    total, count, unchanged = 0, 1, 0
    for i in range(len(pieces)):
        if total + pieces[i] >= min_weight:
            count += 1
            if total == 0:
                unchanged += 1
            total = 0
        else:
            total += pieces[i]
    if count == K + 1 and total == 0:
        unchanged += 1
    return unchanged

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N, K = int(data[0]), int(data[1])
    A = list(map(int, data[2:]))
    
    min_weight = find_min_weight(A + A, K)
    unchanged_cuts = count_unchanged_cuts(A + A, min_weight, K)
    
    print(min_weight, unchanged_cuts - 1)

solve()