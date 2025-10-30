def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    H = list(map(int, data[1:N+1]))
    
    T = 0
    for i in range(N):
        h = H[i]
        # Calculate the number of attacks needed for this enemy
        # Each group of 3 attacks reduces health by 5 (1 + 1 + 3)
        # So, for every 3 attacks, health reduces by 5
        # The remainder is handled separately
        groups = h // 5
        remainder = h % 5
        # Total attacks for this enemy is groups * 3 + (remainder > 0 ? 1 : 0)
        # But need to consider the sequence of attacks
        # For the first enemy, it's straightforward
        # For subsequent enemies, the T value is cumulative
        # So, for each enemy, we need to calculate the T value after processing it
        # Let's compute the T value for this enemy
        # The T value after processing this enemy is T + (groups * 3) + (1 if remainder > 0 else 0)
        # But need to ensure that the attacks are in the correct sequence
        # For example, if remainder is 1, it takes 1 attack
        # If remainder is 2, it takes 2 attacks
        # If remainder is 3, it takes 1 attack (since the third attack is a multiple of 3)
        # If remainder is 4, it takes 2 attacks
        # So, the number of attacks for the remainder is:
        if remainder == 0:
            attacks = groups * 3
        elif remainder == 1:
            attacks = groups * 3 + 1
        elif remainder == 2:
            attacks = groups * 3 + 2
        elif remainder == 3:
            attacks = groups * 3 + 1
        elif remainder == 4:
            attacks = groups * 3 + 2
        T += attacks
    
    print(T)

if __name__ == "__main__":
    main()