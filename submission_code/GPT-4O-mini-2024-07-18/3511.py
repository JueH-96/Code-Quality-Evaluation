class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Calculate the maximum number of turns possible
        turns = 0
        
        while True:
            # Check if Alice can make a move
            if x > 0 and y >= 4:  # 1 coin of 75 and 4 coins of 10
                x -= 1
                y -= 4
            elif y >= 11:  # 11 coins of 10
                y -= 11
            elif x >= 2 and y >= 1:  # 2 coins of 75 and 1 coin of 10
                x -= 2
                y -= 1
            elif x >= 1 and y >= 8:  # 1 coin of 75 and 8 coins of 10
                x -= 1
                y -= 8
            else:
                break  # Alice cannot make a move, she loses
            
            turns += 1
            
            # Check if Bob can make a move
            if x > 0 and y >= 4:  # 1 coin of 75 and 4 coins of 10
                x -= 1
                y -= 4
            elif y >= 11:  # 11 coins of 10
                y -= 11
            elif x >= 2 and y >= 1:  # 2 coins of 75 and 1 coin of 10
                x -= 2
                y -= 1
            elif x >= 1 and y >= 8:  # 1 coin of 75 and 8 coins of 10
                x -= 1
                y -= 8
            else:
                break  # Bob cannot make a move, he loses
            
            turns += 1
        
        # If turns are even, Alice wins, otherwise Bob wins
        return "Alice" if turns % 2 == 0 else "Bob"