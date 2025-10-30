N, M = map(int, input().split())
S = list(input().strip())
T = input().strip()

for k in range(M):
    char_to_place = T[k]
    best_pos = 0
    best_result = None
    
    # Try placing char_to_place at each position
    for i in range(N):
        temp_S = S[:]
        temp_S[i] = char_to_place
        
        # Simulate remaining operations greedily
        for j in range(k + 1, M):
            next_char = T[j]
            # Find leftmost position where next_char improves the string
            placed = False
            for l in range(N):
                if temp_S[l] < next_char:
                    temp_S[l] = next_char
                    placed = True
                    break
            # If no improvement possible, place at rightmost position
            if not placed:
                temp_S[N - 1] = next_char
        
        # Keep track of best result
        if best_result is None or temp_S > best_result:
            best_result = temp_S[:]
            best_pos = i
    
    S[best_pos] = char_to_place

print(''.join(S))