def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    l = int(next(it))
    # Read Takahashi's, Aoki's, and table's cards.
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(m)]
    C = [int(next(it)) for _ in range(l)]
    
    # Total number of cards and create a values array.
    # We'll label cards as follows:
    # 0 .. n-1: Takahashi's initial cards.
    # n .. n+m-1: Aoki's initial cards.
    # n+m .. n+m+l-1: Cards initially on the table.
    total = n + m + l
    values = A + B + C

    # Represent the state as three bit masks (for Takahashi's hand, Aoki's hand, and the table)
    # along with an indicator for whose turn it is: 0 for Takahashi, 1 for Aoki.
    # Initially:
    maskT = (1 << n) - 1  # bits 0..n-1 set.
    maskA = ((1 << m) - 1) << n  # bits n..n+m-1 set.
    maskTable = ((1 << l) - 1) << (n + m)  # bits n+m.. total-1 set.
    
    # We'll use recursion with memoization (via lru_cache) to explore the game states.
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(maskT, maskA, maskTable, player):
        # player: 0 for Takahashi, 1 for Aoki.
        # If the current player has no card in hand, he cannot move and loses.
        if player == 0:
            if maskT == 0:
                return False
            cur_hand = maskT
        else:
            if maskA == 0:
                return False
            cur_hand = maskA
        
        # For each card in the current player's hand, try playing it.
        # When playing a card x:
        #   (1) Remove it from the hand.
        #   (2) Put it on the table (i.e. add it to maskTable).
        #   (3) Optionally pick one card from the table with a value less than the played card.
        # The player may choose not to take any.
        # If any legal move leads to a state from which the opponent loses, then the current state is winning.
        bit = cur_hand
        while bit:
            # Extract one card (lowest set bit in current hand).
            i = (bit & -bit).bit_length() - 1
            bit -= (1 << i)
            newTable = maskTable | (1 << i)  # Place card i onto the table.
            # Option 1: Do not take any card.
            if player == 0:
                new_maskT = maskT & ~(1 << i)
                new_maskA = maskA
            else:
                new_maskA = maskA & ~(1 << i)
                new_maskT = maskT
            # Next turn goes to the opponent.
            if not dp(new_maskT, new_maskA, newTable, 1 - player):
                return True
            
            # Option 2: Choose to take one card (other than the played card) from the table,
            # provided its value is less than that of the played card.
            # Iterate over the cards on the table (newTable) except card i.
            table_mask = newTable & ~(1 << i)
            subset = table_mask
            while subset:
                j = (subset & -subset).bit_length() - 1
                subset -= (1 << j)
                if values[j] < values[i]:
                    # If we take card j, add it to the current player's hand
                    if player == 0:
                        new_maskT2 = (maskT & ~(1 << i)) | (1 << j)
                        new_maskA2 = maskA
                    else:
                        new_maskA2 = (maskA & ~(1 << i)) | (1 << j)
                        new_maskT2 = maskT
                    # Remove card j from the table.
                    newTable2 = newTable & ~(1 << j)
                    if not dp(new_maskT2, new_maskA2, newTable2, 1 - player):
                        return True
        
        # If no move leads to a win, this state is losing.
        return False

    result = dp(maskT, maskA, maskTable, 0)
    sys.stdout.write("Takahashi" if result else "Aoki")

if __name__ == '__main__':
    main()