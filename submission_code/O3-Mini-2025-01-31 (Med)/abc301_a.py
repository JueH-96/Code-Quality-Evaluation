def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    # Parse input
    N = int(input_data[0])
    S = input_data[1]
    
    # Count wins for each player
    total_T = S.count('T')
    total_A = S.count('A')
    
    # If one wins more games, immediately decide the overall winner.
    if total_T > total_A:
        print("T")
        return
    elif total_A > total_T:
        print("A")
        return
    
    # Else: both wins are equal. We simulate the games in order
    cum_T = 0
    cum_A = 0
    for ch in S:
        if ch == 'T':
            cum_T += 1
        else:
            cum_A += 1
        # If one player reaches the final win count earlier than the other.
        if cum_T == total_T and cum_A < total_A:
            print("T")
            return
        if cum_A == total_A and cum_T < total_T:
            print("A")
            return

# Call the main function    
if __name__ == '__main__':
    main()