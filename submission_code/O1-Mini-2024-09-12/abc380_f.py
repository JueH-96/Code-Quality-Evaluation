# YOUR CODE HERE
import sys
import sys
import sys
from functools import lru_cache

def solve():
    import sys
    sys.setrecursionlimit(1000000)
    N, M, L = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    C = list(map(int, sys.stdin.readline().split()))
    
    # Collect all unique values and sort them
    unique_values = sorted(set(A + B + C))
    value_to_index = {v:i for i,v in enumerate(unique_values)}
    num_values = len(unique_values)
    
    # Initialize counts
    ta_hand = [0]*num_values
    for a in A:
        ta_hand[value_to_index[a]] +=1
    ao_hand = [0]*num_values
    for b in B:
        ao_hand[value_to_index[b]] +=1
    table = [0]*num_values
    for c in C:
        table[value_to_index[c]] +=1
        
    unique_values = sorted(unique_values)
    
    @lru_cache(maxsize=None)
    def can_win(ta, ao, tbl, is_takahashi_turn):
        if is_takahashi_turn:
            # Check if Tak has any card to play
            if sum(ta) ==0:
                return False
            # Iterate over possible cards to play
            for i in range(num_values):
                if ta[i] >0:
                    played_card = unique_values[i]
                    # Update Tak's hand and table
                    new_ta = list(ta)
                    new_ta[i] -=1
                    new_tbl = list(tbl)
                    new_tbl[i] +=1
                    # Find eligible cards to take
                    eligible = [j for j in range(num_values) if unique_values[j] < played_card and new_tbl[j] >0]
                    
                    # Option 1: Do not take any card
                    if not can_win(tuple(new_ta), tuple(ao), tuple(new_tbl), False):
                        return True
                    # Option 2: Take any eligible card
                    for j in eligible:
                        tak_take = list(new_ta)
                        taf_take = list(new_tbl)
                        tak_take[j] +=1
                        taf_take[j] -=1
                        if not can_win(tuple(tak_take), tuple(ao), tuple(taf_take), False):
                            return True
            # All moves lead to opponent's win
            return False
        else:
            # Aoki's turn
            if sum(ao) ==0:
                return False
            for i in range(num_values):
                if ao[i] >0:
                    played_card = unique_values[i]
                    # Update Aoki's hand and table
                    new_ao = list(ao)
                    new_ao[i] -=1
                    new_tbl = list(tbl)
                    new_tbl[i] +=1
                    # Find eligible cards to take
                    eligible = [j for j in range(num_values) if unique_values[j] < played_card and new_tbl[j] >0]
                    
                    # Option 1: Do not take any card
                    if not can_win(tuple(ta), tuple(new_ao), tuple(new_tbl), True):
                        return True
                    # Option 2: Take any eligible card
                    for j in eligible:
                        aok_take = list(new_ao)
                        taf_take = list(new_tbl)
                        aok_take[j] +=1
                        taf_take[j] -=1
                        if not can_win(tuple(ta), tuple(aok_take), tuple(taf_take), True):
                            return True
            # All moves lead to opponent's win
            return False
    
    if can_win(tuple(ta_hand), tuple(ao_hand), tuple(table), True):
        print("Takahashi")
    else:
        print("Aoki")