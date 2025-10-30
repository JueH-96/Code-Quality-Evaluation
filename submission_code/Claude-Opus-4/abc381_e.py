def solve():
    N, Q = map(int, input().split())
    S = input().strip()
    
    for _ in range(Q):
        L, R = map(int, input().split())
        # Convert to 0-indexed
        L -= 1
        
        # Extract substring
        substring = S[L:R]
        
        # Find positions of each character
        ones_before = []  # Count of 1s before each position
        twos_after = []   # Count of 2s after each position
        slash_positions = []
        
        # Count 1s before each position
        count_1 = 0
        for i in range(len(substring)):
            ones_before.append(count_1)
            if substring[i] == '1':
                count_1 += 1
        
        # Count 2s after each position
        count_2 = 0
        twos_after = [0] * len(substring)
        for i in range(len(substring) - 1, -1, -1):
            if i < len(substring) - 1:
                twos_after[i] = count_2
            if substring[i] == '2':
                count_2 += 1
        
        # Find slash positions
        for i in range(len(substring)):
            if substring[i] == '/':
                slash_positions.append(i)
        
        # If no slash, answer is 0
        if not slash_positions:
            print(0)
            continue
        
        # Try each slash position
        max_length = 0
        for slash_pos in slash_positions:
            # Count 1s before this slash
            ones = ones_before[slash_pos]
            # Count 2s after this slash
            twos = twos_after[slash_pos]
            
            # Maximum k is minimum of ones and twos
            k = min(ones, twos)
            
            # Length of 11/22 string would be 2k + 1
            if k > 0:
                max_length = max(max_length, 2 * k + 1)
        
        print(max_length)

solve()