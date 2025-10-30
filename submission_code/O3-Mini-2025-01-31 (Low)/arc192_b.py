def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Key idea:
    # In the game each index i is “claimed” the first time a player chooses it.
    # This claim is like a forced move: eventually, every index must be claimed
    # (i.e. moved from 0 free moves available, where "free moves" = A_i - 1).
    # However, before the first move on a given index, there are A_i - 1 moves
    # that can be “wasted” (i.e. moves that do not add i to S because it is still not claimed)
    # by delaying the claim.
    # Since both players can waste moves arbitrarily on already claimed piles,
    # they postpone the moment when an “unclaimed,” forced move (claim move) occurs.
    #
    # Therefore the order of the forced moves (the first move on each pile) 
    # will eventually be determined by the parity of the total number of free moves available:
    # Total free moves = sum_i (A_i - 1) = (sum A) - N.
    #
    # With optimal play Fennec (starting first) can force a win 
    # if and only if (sum A - N) is odd; otherwise Snuke wins.
    #
    # Check sample cases:
    # Sample 1: A=[1,9,2] gives free moves = (1+9+2 - 3)=9, odd -> Fennec.
    # Sample 2: A=[25,29] gives free moves = 25+29-2 = 52, even -> Snuke.
    # Sample 3: A=[1,9,2,25,2,9] gives free moves = 1+9+2+25+2+9 - 6 = 42, even -> Snuke.
    
    total_free = sum(A) - N
    if total_free % 2 == 1:
        sys.stdout.write("Fennec")
    else:
        sys.stdout.write("Snuke")
    
if __name__ == '__main__':
    main()