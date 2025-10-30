import sys

def solve():
    """
    Solves the Fennec vs Snuke game problem.
    """
    try:
        # Fast I/O
        readline = sys.stdin.readline
        
        N_str = readline()
        if not N_str: return
        N = int(N_str)
        
        A = list(map(int, readline().split()))
    except (IOError, ValueError):
        # Handle potential empty lines or invalid format at the end of input
        return

    # Count of undiscovered items that add an even/odd number of delay moves.
    # A_i - 1 is the number of delay moves.
    # A_i - 1 is even if A_i is odd.
    # A_i - 1 is odd if A_i is even.
    
    # E: count of i where A_i is odd (A_i-1 is even)
    # O: count of i where A_i is even (A_i-1 is odd)
    E = 0
    O = 0
    for x in A:
        if x % 2 == 1:
            E += 1
        else:
            O += 1

    if O == 0:
        # Case 1: All A_i are odd. All discoveries add an even number of delay moves.
        # The coin pool parity is always 0. No strategic delaying is possible.
        # The game is a simple race of N moves.
        if N % 2 == 1:
            print("Fennec")
        else:
            print("Snuke")
    elif E == 0:
        # Case 2: All A_i are even. All discoveries add an odd number of delay moves.
        # If N=1, Fennec makes the one move and wins.
        # If N>=2, it can be shown that Snuke has a winning strategy.
        if N == 1:
            print("Fennec")
        else:
            print("Snuke")
    else:
        # Case 3: Mixed parities. E > 0 and O > 0.
        # The game's complexity revolves around the 'O' items, which flip pot parity.
        # If O is odd, the first player (Fennec) can take one, leaving an even number
        # of 'O' items and putting the second player (Snuke) on the back foot.
        if O % 2 == 1:
            print("Fennec")
        else:
        # If O is even, Snuke can mirror Fennec's 'O' moves, neutralizing her advantage.
            print("Snuke")

solve()