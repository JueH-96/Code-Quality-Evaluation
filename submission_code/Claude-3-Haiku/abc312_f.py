from typing import List, Tuple
from collections import defaultdict

def solve(N: int, M: int, items: List[Tuple[int, int]]) -> int:
    # Sort the items by their happiness value in descending order
    items.sort(key=lambda x: x[1], reverse=True)
    
    # Create a dictionary to store the can openers and the number of cans they can open
    can_openers = defaultdict(int)
    
    max_happiness = 0
    
    for t, x in items:
        if t == 0:  # Pull-tab can
            max_happiness += x
            M -= 1
        elif t == 1:  # Regular can
            if M > 0:
                max_happiness += x
                M -= 1
        else:  # Can opener
            # Find the can opener that can open the most cans
            max_can_opener = max(can_openers, key=can_openers.get, default=None)
            if max_can_opener is None or can_openers[max_can_opener] < x:
                can_openers[x] = x
                M -= 1
            else:
                can_openers[max_can_opener] -= 1
                can_openers[x] = x
                M -= 1
        
        if M == 0:
            break
    
    return max_happiness