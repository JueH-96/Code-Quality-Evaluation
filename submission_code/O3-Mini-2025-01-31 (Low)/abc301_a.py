def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    S = input_data[1]
    
    # Count total wins for each player
    total_T = S.count('T')
    total_A = S.count('A')
    
    # Determine if one player won more games overall
    if total_T > total_A:
        print("T")
        return
    elif total_A > total_T:
        print("A")
        return
    # Otherwise, the total wins are tied: total_T == total_A
    # We need to find who reached that number of wins first.
    count_T = 0
    count_A = 0
    target = total_T  # since total_T == total_A
    for ch in S:
        if ch == 'T':
            count_T += 1
        else:  # ch == 'A'
            count_A += 1
        # Check if one has reached the target wins.
        if count_T == target:
            print("T")
            return
        if count_A == target:
            print("A")
            return

if __name__ == '__main__':
    main()