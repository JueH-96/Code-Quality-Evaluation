def count_fine_triplets(S):
    S_set = set(S)
    count = 0
    
    # Sort S to ensure A < C
    S_sorted = sorted(S)
    
    for i, A in enumerate(S_sorted):
        for j in range(i+1, len(S_sorted)):
            C = S_sorted[j]
            # If (A + C) is even, then B is an integer
            if (A + C) % 2 == 0:
                B = (A + C) // 2
                if B in S_set and A < B < C:
                    count += 1
                    
    return count

N = int(input())
S = list(map(int, input().split()))
print(count_fine_triplets(S))