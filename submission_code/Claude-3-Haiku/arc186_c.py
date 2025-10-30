from typing import List

def solve_test_case(N: int, M: int, boxes: List[List[int]]) -> int:
    """
    Solves a single test case of the game between Mr. Ball and Mr. Box.
    
    Args:
        N (int): The number of boxes.
        M (int): The number of types of balls.
        boxes (List[List[int]]): A list of [capacity, price] for each box.
    
    Returns:
        int: The increase in Mr. Box's money.
    """
    # Sort the boxes by their capacity
    boxes.sort(key=lambda x: x[0])
    
    # Initialize the number of balls of each type in each box
    ball_counts = [[0] * M for _ in range(N)]
    
    # Initialize Mr. Box's money
    money = 0
    
    # Play the game
    for _ in range(10 ** 100 * M):
        # Mr. Ball chooses a ball and gives it to Mr. Box
        ball_type = _ % M
        
        # Find the first box that can accept the ball
        for i in range(N):
            if ball_counts[i][ball_type] < boxes[i][0]:
                # Put the ball in the box
                ball_counts[i][ball_type] += 1
                
                # Check if the box satisfies the conditions
                if all(count == 0 or count == ball_counts[i][ball_type] for count in ball_counts[i]):
                    # Mr. Box receives 1 yen
                    money += 1
                break
        else:
            # Mr. Box chooses to end the game
            break
    
    return money

def main():
    # Read the input
    T = int(input())
    for _ in range(T):
        N, M = map(int, input().split())
        boxes = [list(map(int, input().split())) for _ in range(N)]
        
        # Solve the test case
        result = solve_test_case(N, M, boxes)
        print(result)

if __name__ == "__main__":
    main()