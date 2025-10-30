def find_kth_character(S, K):
    n = len(S)
    pos = (K - 1) % n
    shift = (K - 1) // n
    
    # Count the number of set bits in the binary representation of shift
    count = bin(shift).count('1')
    
    char = S[pos]
    
    # If count is odd, invert the case of the character
    if count % 2 == 1:
        if char.islower():
            return char.upper()
        else:
            return char.lower()
    else:
        return char

def solve():
    S = input().strip()
    Q = int(input().strip())
    K_list = list(map(int, input().strip().split()))
    
    results = []
    for K in K_list:
        results.append(find_kth_character(S, K))
    
    print(' '.join(results))

solve()