# YOUR CODE HERE
def main():
    S = input().strip()
    N = int(input().strip())
    L = len(S)
    
    # Compute minimal_num by setting '?' to '0' and '1's as is
    minimal_num = 0
    for i in range(L):
        if S[i] == '1':
            bit_pos = L -1 -i
            minimal_num += (1 << bit_pos)
    
    if minimal_num > N:
        print(-1)
        return
    
    # Precompute max_add array
    max_add = [0] * (L +1)
    for i in range(L-1, -1, -1):
        if S[i] == '1' or S[i] == '?':
            bit_pos = L -1 -i
            max_add[i] = max_add[i+1] + (1 << bit_pos)
        else:
            max_add[i] = max_add[i+1]
    
    current_num =0
    for i in range(L):
        bit_pos = L -1 -i
        char = S[i]
        if char == '0':
            continue
        elif char == '1':
            current_num += (1 << bit_pos)
        elif char == '?':
            tent_num = current_num + (1 << bit_pos)
            max_possible = tent_num + max_add[i+1]
            if max_possible <= N:
                current_num = tent_num
            # else set to '0', do nothing
    if current_num <= N:
        print(current_num)
    else:
        print(-1)

if __name__ == "__main__":
    main()