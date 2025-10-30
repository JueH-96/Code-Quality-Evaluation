# YOUR CODE HERE
N, M = map(int, input().split())
S = list(input().strip())
T = input().strip()

for k in range(M):
    t_char = T[k]
    
    # Find the leftmost position where we can improve (t_char > s_char)
    best_pos = -1
    for i in range(N):
        if t_char > S[i]:
            best_pos = i
            break
    
    # If we found an improvement position, use it
    if best_pos != -1:
        S[best_pos] = t_char
    else:
        # If no improvement possible, replace the rightmost position
        S[N-1] = t_char

print(''.join(S))