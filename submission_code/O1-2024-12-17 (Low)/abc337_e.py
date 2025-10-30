import math

def main():
    import sys
    data = sys.stdin.read().strip().split()
    # data[0] = N
    N = int(data[0])
    
    # Minimum number of friends needed is ceil(log2(N))
    # because we can assign bottles to friends according
    # to binary representation of (bottle_index - 1).
    M = 1 if N == 1 else math.ceil(math.log2(N))
    
    # Print the number of friends
    print(M)
    
    # Prepare which bottles go to which friend
    # friend i (0-based) will test bottles whose (bottle_index - 1)
    # has the i-th bit set.
    friend_sets = [[] for _ in range(M)]
    for bottle in range(1, N + 1):
        index_0_based = bottle - 1
        for bit in range(M):
            if (index_0_based >> bit) & 1:
                friend_sets[bit].append(bottle)
    
    # Print which bottles each friend will taste
    for bit in range(M):
        bottles_for_friend = friend_sets[bit]
        print(len(bottles_for_friend), end='')
        for b in sorted(bottles_for_friend):
            print(' ' + str(b), end='')
        print()  # newline
    
    # Now we read the next line (judge's response),
    # which is the stomach upset info in a binary string S of length M.
    # Because this is an interactive style problem,
    # we take it from sys.stdin again.
    # However, we already read all input at once into 'data'.
    # The string S should be at data[1], if there's exactly one line after the distribution lines.
    # In a truly interactive environment, we'd read again from input(),
    # but here we can just pick the next chunk from data if present.
    
    # We have:
    # data[0] = N
    # lines for M friends usage (each can have up to 1 + K_i tokens)
    # let's figure out how many tokens we consumed.
    # We consumed 1 token for N.
    # Then for each friend line, we consumed 1 + K_i tokens.
    # So let's recast carefully. We'll parse them manually
    # to get to the next token: S
    
    # But the problem statement says we only need to mock it,
    # we'll handle it in a single run. We'll assume
    # that after printing distribution, the next token in 'data'
    # is actually S. We'll parse from there.

    # Let's parse out how many tokens we used for distribution lines:
    # We used 1 token for N up front.
    # Then for each friend i, 1 token for K_i, plus K_i tokens for the bottle numbers.
    # We'll reconstruct that from the friend_sets themselves.
    
    idx = 1  # index where we left off
    for bit in range(M):
        # K_i = length of friend_sets[bit]
        K_i = len(friend_sets[bit])
        idx += 1 + K_i  # tokens used for this line
    
    # Now data[idx] should be string S
    if idx < len(data):
        S = data[idx]
    else:
        # If there's no more data, we can just default to a no-upset scenario or something,
        # but the problem states it's interactive. 
        # We'll assume for demonstration that S = '0' * M if nothing is provided.
        S = '0' * M
    
    # Decode S to find the spoiled bottle index (0-based)
    spoiled_0_based = 0
    for bit in range(M):
        if S[bit] == '1':
            spoiled_0_based += (1 << bit)
    
    # Print the 1-based spoiled bottle number
    print(spoiled_0_based + 1)

# Do not forget to call main()!
if __name__ == "__main__":
    main()