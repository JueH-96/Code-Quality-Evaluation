# YOUR CODE HERE
import sys
import math
from collections import deque

def get_product(s):
    """Evaluates the product of a string expression like '1*2*3'."""
    if not s:
        return 1
    parts = s.split('*')
    prod = 1
    for part in parts:
        prod *= int(part)
    return prod

def solve():
    """
    Main solver function.
    Reads input N and searches for a palindromic expression S.
    """
    N = int(sys.stdin.readline())

    # Case 1: S is just a number.
    sN = str(N)
    if '0' not in sN and sN == sN[::-1]:
        print(sN)
        return

    # Case 2: S has a composite structure.
    # We perform a Breadth-First Search (BFS) on the possible left parts of the string.
    
    queue = deque()
    
    # Initialize queue with single numbers
    # A reasonable upper bound for the first number. If d*rev(d) > N, we can often stop.
    # d*d is a rough lower bound for d*rev(d), so d <= sqrt(N) is a good heuristic.
    # limit = int(math.sqrt(N)) + 2 
    # The actual length of the left part turns out to be very small, so a smaller
    # limit for the initial numbers is fine.
    limit = 100000 
    for i in range(1, limit):
        s_i = str(i)
        if '0' in s_i:
            continue
        queue.append((s_i, i))
    
    visited = set()

    while queue:
        left_s, left_prod = queue.popleft()

        if left_s in visited:
            continue
        visited.add(left_s)

        # Check this left_s as a complete left part.
        # The structure of S could be L*C*R or L*R.
        right_s = left_s[::-1]
        
        try:
            right_prod = get_product(right_s)
        except (ValueError, IndexError):
            continue

        if left_prod > N:
            continue
        
        # Using integer arithmetic to avoid floating point inaccuracies and overflow
        if right_prod == 0 or left_prod > N // right_prod:
            continue

        prod_pair = left_prod * right_prod
        
        if N % prod_pair == 0:
            C = N // prod_pair
            sC = str(C)
            if '0' not in sC and sC == sC[::-1]:
                # Found a potential solution
                if C == 1:
                    # Structure is L * R
                    # If L is a single number, R=L, S=L*L.
                    # If L=d1*d2, R=rev(d2)*rev(d1), S=d1*d2*rev(d2)*rev(d1) etc.
                    # The formula is L + '*' + R
                    S = left_s + '*' + right_s
                else:
                    # Structure is L * C * R
                    S = left_s + '*' + sC + '*' + right_s
                
                if len(S) <= 1000:
                    print(S)
                    return

        # Extend the left part and add to queue
        # Heuristic bound to keep search space manageable
        if len(left_s) >= 15:
            continue
        
        # Append another number to the left_s
        # A small bound like 1000 is sufficient for the next number.
        for i in range(1, 1000):
            s_i = str(i)
            if '0' in s_i:
                continue
            
            new_left_s = left_s + '*' + s_i
            
            if len(new_left_s) > 15:
                break
                
            new_left_prod = left_prod * i
            
            if new_left_prod > N:
                break
            
            if new_left_s not in visited:
                queue.append((new_left_s, new_left_prod))

    print(-1)

if __name__ == "__main__":
    solve()