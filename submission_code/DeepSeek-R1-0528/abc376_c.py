import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    B = list(map(int, data[1+n:1+n+n-1]))
    
    A.sort()
    B.sort()
    j = 0
    pairs = []
    for box in B:
        if j < len(A) and A[j] <= box:
            pairs.append((A[j], box))
            j += 1
            
    if j < n-1:
        print(-1)
        return
        
    unmatched_toy = A[j]
    candidate1 = unmatched_toy
    
    M = float('inf')
    for a, box_size in pairs:
        if box_size >= candidate1:
            if a < M:
                M = a
                
    candidate2 = float('inf')
    for a, box_size in pairs:
        if box_size >= candidate1:
            candidate2 = min(candidate2, a)
        else:
            if M <= box_size:
                candidate2 = min(candidate2, a)
                
    ans = min(candidate1, candidate2)
    print(ans)

if __name__ == '__main__':
    main()