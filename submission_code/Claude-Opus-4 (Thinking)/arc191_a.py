N, M = map(int, input().split())
S = list(input().strip())
T = input().strip()

for k in range(M):
    char = T[k]
    
    # Check if there's a larger character coming later
    has_larger = False
    for j in range(k+1, M):
        if T[j] > char:
            has_larger = True
            break
    
    if has_larger:
        # Save leftmost positions for larger characters
        # Place char at rightmost position where it improves (or just rightmost)
        best_pos = -1
        for i in range(N-1, -1, -1):
            if S[i] < char:
                best_pos = i
                break
        if best_pos != -1:
            S[best_pos] = char
        else:
            S[N-1] = char
    else:
        # No larger character coming, place greedily
        best_pos = -1
        for i in range(N):
            if S[i] < char:
                best_pos = i
                break
        if best_pos != -1:
            S[best_pos] = char
        else:
            S[N-1] = char

print(''.join(S))