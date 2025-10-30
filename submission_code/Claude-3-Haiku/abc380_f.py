from typing import List

def solve(N: int, M: int, L: int, A: List[int], B: List[int], C: List[int]) -> str:
    """
    Determines the winner of the game between Takahashi and Aoki.
    
    Args:
        N (int): The number of cards Takahashi has.
        M (int): The number of cards Aoki has.
        L (int): The number of cards on the table.
        A (List[int]): The numbers on Takahashi's cards.
        B (List[int]): The numbers on Aoki's cards.
        C (List[int]): The numbers on the cards on the table.
    
    Returns:
        str: "Takahashi" if Takahashi wins, "Aoki" if Aoki wins.
    """
    # Sort the cards on the table in ascending order
    C.sort()
    
    # Takahashi's turn
    def takahashi_turn(takahashi_hand: List[int], aoki_hand: List[int], table: List[int]) -> bool:
        if not takahashi_hand:
            return False
        
        # Choose a card from Takahashi's hand and put it on the table
        card = takahashi_hand.pop()
        table.append(card)
        
        # Check if Takahashi can take a card from the table
        for i, c in enumerate(table):
            if c < card:
                takahashi_hand.append(c)
                table.pop(i)
                break
        
        return True
    
    # Aoki's turn
    def aoki_turn(takahashi_hand: List[int], aoki_hand: List[int], table: List[int]) -> bool:
        if not aoki_hand:
            return False
        
        # Choose a card from Aoki's hand and put it on the table
        card = aoki_hand.pop()
        table.append(card)
        
        # Check if Aoki can take a card from the table
        for i, c in enumerate(table):
            if c < card:
                aoki_hand.append(c)
                table.pop(i)
                break
        
        return True
    
    # Initialize the game state
    takahashi_hand = A
    aoki_hand = B
    table = C
    
    # Play the game
    while True:
        if not takahashi_turn(takahashi_hand, aoki_hand, table):
            return "Aoki"
        if not aoki_turn(takahashi_hand, aoki_hand, table):
            return "Takahashi"

# Read input from stdin
N, M, L = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
C = list(map(int, input().split()))

# Solve the problem and print the answer to stdout
print(solve(N, M, L, A, B, C))